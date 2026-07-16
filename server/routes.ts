import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { WebSocketServer, WebSocket } from "ws";
import AdmZip from "adm-zip";
import { parse } from "node-html-parser";
import { ProcessingCancelledError, textProcessor } from "./text-processor";
import { llmService, setHuggingFaceApiToken, getHuggingFaceApiToken } from "./llm-service";
import { nanoid } from "nanoid";
import type { ProcessingConfig, LogEntry, WSMessage, HuggingFaceTokenStatus } from "@shared/schema";
import {
  huggingFaceTokenUpdateSchema,
  deterministicCleanRequestSchema,
  processTextRequestSchema,
  speechEngineSchema,
  speechExecutionTargetSchema,
} from "@shared/schema";
import { clampCharacterSampleSize, getCharacterSampleCeiling } from "@shared/model-utils";
import { chunkTextBySentences, countWords, segmentSentences } from "@shared/text-utils";
import { indexTtsService } from "./tts-service";
import { vibevoiceService } from "./vibevoice-service";
import { pdfOcrService } from "./pdf-ocr-service";
import { speechService } from "./speech-service";
import { huggingFaceUsageService } from "./huggingface-usage-service";
import { registerVoiceForgeMcpRoutes, voiceForgeMcpHealth } from "./mcp-http";
import { voiceLibraryService } from "./voice-library-service";
import fs from "fs";
import path from "path";
import { applyDeterministicCleaning } from "./text-cleaner";
import { evaluateLocalRequest } from "./loopback-request-policy";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 1,
  },
});

const ttsUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 32 * 1024 * 1024,
    files: 2,
    fieldSize: 2 * 1024 * 1024,
    fields: 8,
  },
});

const vibevoiceUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 24 * 1024 * 1024,
    files: 5,
    fieldSize: 2 * 1024 * 1024,
    fields: 12,
  },
});

const pdfUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 64 * 1024 * 1024,
    files: 1,
  },
});

const speechUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 32 * 1024 * 1024,
    files: 2,
    fieldSize: 2 * 1024 * 1024,
    fields: 32,
  },
});

const MAX_EPUB_ENTRIES = 5_000;
const MAX_EPUB_ENTRY_BYTES = 10 * 1024 * 1024;
const MAX_EPUB_EXPANDED_BYTES = 50 * 1024 * 1024;
const MAX_EXTRACTED_TEXT_CHARS = 5_000_000;
const MAX_EXTRACTED_TEXT_BYTES = 5 * 1024 * 1024;
const MAX_SYNTHESIS_TEXT_CHARS = 500_000;
const ALLOWED_AUDIO_EXTENSIONS = new Set([
  ".wav",
  ".mp3",
  ".flac",
  ".m4a",
  ".aac",
  ".ogg",
  ".opus",
  ".webm",
]);

class RequestInputError extends Error {}

function decodeTextBuffer(buffer: Buffer): string {
  let text: string;
  if (buffer[0] === 0xff && buffer[1] === 0xfe) {
    text = new TextDecoder("utf-16le", { fatal: true }).decode(buffer);
  } else if (buffer[0] === 0xfe && buffer[1] === 0xff) {
    text = new TextDecoder("utf-16be", { fatal: true }).decode(buffer);
  } else {
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
    } catch {
      text = new TextDecoder("windows-1252", { fatal: true }).decode(buffer);
    }
  }
  text = text.replace(/^\uFEFF/, "");
  if (text.includes("\u0000")) throw new RequestInputError("The text file appears to contain binary data");
  return text;
}

function validateAudioUpload(file: Express.Multer.File): void {
  const extension = path.extname(path.basename(file.originalname)).toLowerCase();
  if (!ALLOWED_AUDIO_EXTENSIONS.has(extension)) {
    throw new RequestInputError("Voice references must be WAV, MP3, FLAC, M4A, AAC, OGG, Opus, or WebM audio");
  }
  if (file.buffer.length === 0) throw new RequestInputError("Voice reference is empty");
}

type ResolvedVoiceInput = {
  buffer: Buffer;
  originalname: string;
  transcript?: string;
};

async function resolveVoiceInput(
  uploaded: Express.Multer.File | undefined,
  rawVoiceId: unknown
): Promise<ResolvedVoiceInput | undefined> {
  const voiceId = typeof rawVoiceId === "string" ? rawVoiceId.trim() : "";
  if (uploaded && voiceId) {
    throw new RequestInputError("Choose either an uploaded reference or a default voice, not both");
  }
  if (uploaded) {
    validateAudioUpload(uploaded);
    return { buffer: uploaded.buffer, originalname: uploaded.originalname };
  }
  if (!voiceId) return undefined;
  const voice = await voiceLibraryService.readVoice(voiceId);
  if (!voice) throw new RequestInputError("The selected default voice is unavailable");
  return {
    buffer: voice.audio,
    originalname: `${voice.metadata.name}.${voice.metadata.format}`,
    transcript: voice.metadata.transcript ?? undefined,
  };
}

function assertSynthesisTextSize(text: string): void {
  if (text.length > MAX_SYNTHESIS_TEXT_CHARS) {
    throw new RequestInputError("Synthesis text exceeds the 500,000 character limit; split it into smaller jobs");
  }
}

function createRequestAbortContext(req: Request, res: Response) {
  const controller = new AbortController();
  const abortIfResponseIncomplete = () => {
    if (!res.writableEnded) controller.abort();
  };

  req.once("aborted", abortIfResponseIncomplete);
  res.once("close", abortIfResponseIncomplete);
  if (req.aborted || res.destroyed) abortIfResponseIncomplete();

  return {
    signal: controller.signal,
    cleanup: () => {
      req.off("aborted", abortIfResponseIncomplete);
      res.off("close", abortIfResponseIncomplete);
    },
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function decodeEpubPath(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function extractEpubDocumentText(content: string): string {
  const root = parse(content);
  const body = root.querySelector("body");
  if (!body) return "";

  body.querySelectorAll("script, style, nav, svg").forEach((node) => node.remove());
  return body.structuredText
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export async function parseEpub(buffer: Buffer): Promise<string> {
  try {
    const zip = new AdmZip(buffer);
    const entries = zip.getEntries();
    if (entries.length > MAX_EPUB_ENTRIES) {
      throw new Error(`EPUB contains too many archive entries (maximum ${MAX_EPUB_ENTRIES.toLocaleString()})`);
    }

    let expandedBytes = 0;
    for (const entry of entries) {
      if (entry.isDirectory) continue;
      const entryBytes = Number(entry.header.size);
      if (!Number.isSafeInteger(entryBytes) || entryBytes < 0) {
        throw new Error(`EPUB contains an invalid archive entry: ${entry.entryName}`);
      }
      if (entryBytes > MAX_EPUB_ENTRY_BYTES) {
        throw new Error(`EPUB entry is too large: ${entry.entryName}`);
      }
      expandedBytes += entryBytes;
      if (expandedBytes > MAX_EPUB_EXPANDED_BYTES) {
        throw new Error("EPUB expands beyond the 50 MB safety limit");
      }
    }

    const entryByName = new Map(entries.map((entry) => [entry.entryName.toLowerCase(), entry]));

    const orderedNames: string[] = [];
    const containerEntry = entryByName.get("meta-inf/container.xml");
    if (containerEntry) {
      const container = parse(containerEntry.getData().toString("utf-8"));
      const opfPath = container.querySelector("rootfile")?.getAttribute("full-path");
      const opfEntry = opfPath ? entryByName.get(decodeEpubPath(opfPath).toLowerCase()) : undefined;

      if (opfEntry && opfPath) {
        const opf = parse(opfEntry.getData().toString("utf-8"));
        const manifest = new Map<string, string>();
        opf.querySelectorAll("manifest item, item").forEach((item) => {
          const id = item.getAttribute("id");
          const href = item.getAttribute("href")?.split("#", 1)[0];
          if (id && href) manifest.set(id, href);
        });

        const opfDirectory = path.posix.dirname(opfPath.replace(/\\/g, "/"));
        opf.querySelectorAll("spine itemref, itemref").forEach((itemRef) => {
          const href = manifest.get(itemRef.getAttribute("idref") ?? "");
          if (!href) return;
          const resolved = path.posix.normalize(path.posix.join(opfDirectory, decodeEpubPath(href)));
          if (!orderedNames.includes(resolved)) orderedNames.push(resolved);
        });
      }
    }

    // Malformed/legacy EPUBs sometimes omit their container or spine. Keep a
    // deterministic archive-order fallback instead of rejecting readable books.
    if (orderedNames.length === 0) {
      entries.forEach((entry) => {
        const name = entry.entryName;
        const lower = name.toLowerCase();
        if (/\.(?:x?html?)$/.test(lower) && !/(?:^|\/)(?:nav|toc)\./.test(lower)) {
          orderedNames.push(name);
        }
      });
    }

    const textParts: string[] = [];
    let extractedChars = 0;
    for (const name of orderedNames) {
      const entry = entryByName.get(name.toLowerCase());
      if (!entry) continue;
      try {
        const text = extractEpubDocumentText(entry.getData().toString("utf-8"));
        if (text) {
          extractedChars += text.length;
          if (extractedChars > MAX_EXTRACTED_TEXT_CHARS) {
            throw new Error("Extracted EPUB text exceeds the 5,000,000 character safety limit");
          }
          textParts.push(text);
        }
      } catch (error) {
        if (error instanceof Error && error.message.includes("safety limit")) throw error;
        console.warn(`Skipping unreadable EPUB document ${entry.entryName}:`, error);
      }
    }

    if (textParts.length === 0) throw new Error("No readable content found in EPUB file");
    return textParts.join("\n\n");
  } catch (error) {
    console.error("EPUB parsing error:", error);
    throw new Error(`Failed to parse EPUB file: ${error instanceof Error ? error.message : "Unknown error"}`);
  }
}

function estimatePdfPageCount(buffer: Buffer): number {
  try {
    const text = buffer.toString("latin1");
    const matches = text.match(/\/Type\s*\/Page\b/g);
    if (matches && matches.length > 0) {
      return matches.length;
    }
  } catch (error) {
    console.error("Failed to estimate PDF page count:", error);
  }
  return 1;
}

function createTokenPreview(token: string): string {
  if (token.length <= 8) {
    return `${token.slice(0, Math.min(4, token.length))}…`;
  }
  return `${token.slice(0, 4)}…${token.slice(-4)}`;
}

function buildTokenStatus(): HuggingFaceTokenStatus {
  const token = getHuggingFaceApiToken();
  return {
    configured: Boolean(token),
    tokenPreview: token ? createTokenPreview(token) : undefined,
  };
}

const HUGGING_FACE_TOKEN_KEYS = ["HUGGINGFACE_API_TOKEN", "HF_TOKEN"] as const;
let tokenPersistenceQueue: Promise<void> = Promise.resolve();

async function replaceEnvValues(
  filePath: string,
  values: Readonly<Record<(typeof HUGGING_FACE_TOKEN_KEYS)[number], string | undefined>>
): Promise<void> {
  const exists = fs.existsSync(filePath);
  if (!exists && HUGGING_FACE_TOKEN_KEYS.every((key) => !values[key])) {
    return;
  }

  const raw = exists ? await fs.promises.readFile(filePath, "utf-8") : "";
  const lines = raw.length > 0 ? raw.split(/\r?\n/) : [];
  const output: string[] = [];

  for (const line of lines) {
    const match = line.match(/^\s*(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (match && HUGGING_FACE_TOKEN_KEYS.includes(match[2] as (typeof HUGGING_FACE_TOKEN_KEYS)[number])) continue;
    output.push(line);
  }

  while (output.length > 0 && output[output.length - 1].trim() === "") {
    output.pop();
  }
  for (const key of HUGGING_FACE_TOKEN_KEYS) {
    const value = values[key];
    if (value) output.push(`${key}=${value}`);
  }

  const content = output.join("\n");
  const temporaryPath = `${filePath}.${process.pid}.${nanoid(8)}.tmp`;
  try {
    await fs.promises.writeFile(temporaryPath, content ? `${content}\n` : "", {
      encoding: "utf-8",
      mode: 0o600,
    });
    await fs.promises.rename(temporaryPath, filePath);
  } finally {
    await fs.promises.rm(temporaryPath, { force: true }).catch(() => undefined);
  }
}

async function persistHuggingFaceToken(token: string | undefined): Promise<void> {
  const envPath = path.resolve(process.cwd(), ".env");
  const envTxtPath = path.resolve(process.cwd(), "env.txt");
  const values = { HUGGINGFACE_API_TOKEN: token, HF_TOKEN: token };
  const operation = tokenPersistenceQueue
    .catch(() => undefined)
    .then(async () => {
      await Promise.all([
        replaceEnvValues(envPath, values),
        replaceEnvValues(envTxtPath, values),
      ]);
    });
  tokenPersistenceQueue = operation.catch(() => undefined);
  await operation;
}

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  registerVoiceForgeMcpRoutes(app);

  app.get("/api/voiceforge/health", (req, res) => {
    const nonce = typeof req.query.nonce === "string" ? req.query.nonce : undefined;
    res.json(voiceForgeMcpHealth(nonce));
  });

  app.get("/api/settings/huggingface-token", (_req, res) => {
    res.json(buildTokenStatus());
  });

  app.get("/api/huggingface/usage", async (_req, res) => {
    try {
      res.json(await huggingFaceUsageService.getStatus());
    } catch (error) {
      console.error("Failed to load Hugging Face usage:", error);
      res.status(502).json({
        error: error instanceof Error ? error.message : "Failed to load Hugging Face usage",
      });
    }
  });

  app.get("/api/voices", async (_req, res) => {
    const voices = await voiceLibraryService.listVoices();
    res.json({
      voices: voices.map((voice) => ({
        id: voice.id,
        displayName: voice.name,
        format: voice.format,
        sizeBytes: voice.sizeBytes,
        hasTranscript: voice.hasTranscript,
        transcript: voice.transcript ?? undefined,
      })),
      warnings: [],
    });
  });

  app.get("/api/voices/:id/audio", async (req, res) => {
    const voice = await voiceLibraryService.readVoice(req.params.id);
    if (!voice) return res.status(404).json({ error: "Default voice not found" });
    const safeName = voice.metadata.name.replace(/["\\\r\n]/gu, "_");
    res.type(voice.metadata.mimeType);
    res.setHeader(
      "Content-Disposition",
      `inline; filename="${safeName}.${voice.metadata.format}"`
    );
    res.send(voice.audio);
  });

  app.post("/api/settings/huggingface-token", async (req, res) => {
    const parsed = huggingFaceTokenUpdateSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: "Invalid payload" });
    }

    const value = parsed.data.token;
    const trimmed = typeof value === "string" ? value.trim() : "";
    const nextToken = trimmed.length > 0 ? trimmed : undefined;

    try {
      await persistHuggingFaceToken(nextToken);
      setHuggingFaceApiToken(nextToken);
      huggingFaceUsageService.invalidate();
      res.json(buildTokenStatus());
    } catch (error) {
      console.error("Failed to update HuggingFace token:", error);
      res.status(500).json({
        error: "Failed to update HuggingFace token",
      });
    }
  });

  app.post("/api/text/clean", (req, res) => {
    const parsed = deterministicCleanRequestSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: "Invalid payload" });
    }

    try {
      const { text, options } = parsed.data;
      const pre = applyDeterministicCleaning(text, options, "pre");
      const post = applyDeterministicCleaning(pre.text, options, "post");
      const appliedSteps = Array.from(new Set([...pre.applied, ...post.applied]));
      res.json({
        cleanedText: post.text,
        appliedSteps,
      });
    } catch (error) {
      console.error("Deterministic cleaning failed:", error);
      res.status(500).json({
        error: "Failed to clean text",
        details: error instanceof Error ? error.message : undefined,
      });
    }
  });

  // Good models list (JSON with pricing preferred; fallback to TXT)
  app.get("/api/good-models", async (_req, res) => {
    try {
      const jsonPath = path.resolve(process.cwd(), "good_models.json");
      if (fs.existsSync(jsonPath)) {
        const raw = await fs.promises.readFile(jsonPath, "utf-8");
        const parsed = JSON.parse(raw);
        const models = Array.isArray(parsed?.models)
          ? parsed.models
          : Array.isArray(parsed)
            ? parsed
            : [];
        return res.json({ models });
      }
      // Fallback to simple text list
      const txtPath = path.resolve(process.cwd(), "good_models.txt");
      if (fs.existsSync(txtPath)) {
        const raw = await fs.promises.readFile(txtPath, "utf-8");
        const lines = raw
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter((l) => l.length > 0 && !l.startsWith("#"));
        const models = lines.map((id) => ({ id }));
        return res.json({ models });
      }
      return res.json({ models: [] });
    } catch (err) {
      console.error("Failed to read good_models.txt:", err);
      return res.status(500).json({ error: "Failed to read good models list" });
    }
  });

  // Prompt preview endpoint
  app.post("/api/preview-prompts", async (req, res) => {
    try {
      const { sampleText, config } = req.body as {
        sampleText: string;
        config: ProcessingConfig;
      };

      if (!sampleText || !config) {
        return res.status(400).json({ error: "Missing sample text or configuration" });
      }

      const prompts = llmService.getPromptPreviews(
        sampleText,
        config.cleaningOptions,
        config.speakerConfig,
        config.customInstructions,
        (config as any).singlePass === true,
        (config as any).extendedExamples === true,
        (config as any).llmCleaningDisabled === true
      );

      res.json(prompts);
    } catch (error) {
      console.error("Preview error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to generate preview",
      });
    }
  });

  // Test one chunk endpoint
  app.post("/api/test-chunk", async (req, res) => {
    const requestAbort = createRequestAbortContext(req, res);
    try {
      const { text, config } = req.body as {
        text: string;
        config: ProcessingConfig;
      };

      if (!text || !config) {
        return res.status(400).json({ error: "Missing text or configuration" });
      }

      // Process just the first chunk (up to batchSize sentences)
      const sentences = segmentSentences(text);
      const testChunk = sentences.slice(0, config.batchSize).join(" ");

      const result = await llmService.processChunk({
        text: testChunk,
        cleaningOptions: config.cleaningOptions,
        speakerConfig: config.speakerConfig,
        modelSource: config.modelSource,
        modelName: config.modelName,
        ollamaModelName: (config as any).ollamaModelName,
        temperature: (config as any).temperature,
        llmCleaningDisabled: (config as any).llmCleaningDisabled === true,
        customInstructions: config.customInstructions,
        singlePass: config.singlePass === true,
        extendedExamples: (config as any).extendedExamples === true,
        signal: requestAbort.signal,
      });

      requestAbort.signal.throwIfAborted();
      res.json({
        originalChunk: testChunk,
        processedChunk: result.text,
        sentenceCount: Math.min(sentences.length, config.batchSize),
        usage: result.usage,
      });
    } catch (error) {
      if (requestAbort.signal.aborted || isAbortError(error)) return;
      if (res.headersSent || res.destroyed) return;
      console.error("Test chunk error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to test chunk",
      });
    } finally {
      requestAbort.cleanup();
    }
  });

  // Extract character names endpoint
  app.post("/api/extract-characters", async (req, res) => {
    const requestAbort = createRequestAbortContext(req, res);
    try {
      const { text, sampleSize, includeNarrator, modelSource, modelName, ollamaModelName, temperature } = req.body as {
        text: string;
        sampleSize: number;
        includeNarrator: boolean;
        modelSource?: string;
        modelName: string;
        ollamaModelName?: string;
        temperature?: number;
      };

      if (!text || !sampleSize) {
        return res.status(400).json({ error: "Missing text or sample size" });
      }

      const resolvedSource = ((modelSource as any) || 'api') as 'api' | 'ollama';
      const maxSample = getCharacterSampleCeiling(resolvedSource, ollamaModelName);
      const effectiveSampleSize = clampCharacterSampleSize(sampleSize, resolvedSource, ollamaModelName);

      // Extract sample from text
      const sentences = segmentSentences(text);
      const sampleText = sentences.slice(0, effectiveSampleSize).join(" ");

      // Use LLM to extract character names and narrator identity
      const { characters, narratorCharacterName } = await llmService.extractCharacters({
        text: sampleText,
        includeNarrator,
        modelSource: resolvedSource,
        modelName: modelName || "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ollamaModelName,
        temperature: (typeof temperature === 'number' && Number.isFinite(temperature)) ? temperature : undefined,
        signal: requestAbort.signal,
      });

      requestAbort.signal.throwIfAborted();
      res.json({
        characters,
        narratorCharacterName,
        sampleSentenceCount: Math.min(sentences.length, effectiveSampleSize),
        sampleLimit: maxSample,
      });
    } catch (error) {
      if (requestAbort.signal.aborted || isAbortError(error)) return;
      if (res.headersSent || res.destroyed) return;
      console.error("Character extraction error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to extract characters",
      });
    } finally {
      requestAbort.cleanup();
    }
  });

  // List installed Ollama models for dropdown pre-population
  app.get("/api/ollama/models", async (_req, res) => {
    try {
      const base = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
      const url = `${base}/api/tags`;
      const r = await fetch(url, { method: 'GET' });
      if (!r.ok) {
        return res.status(502).json({ error: `Failed to query Ollama (${r.status})` });
      }
      const data: any = await r.json();
      const models = Array.isArray(data?.models)
        ? data.models.map((m: any) => ({ id: String(m?.name || m?.model || '').trim() })).filter((m: any) => m.id)
        : [];
      return res.json({ models });
    } catch (e) {
      return res.status(500).json({ error: 'Failed to list Ollama models' });
    }
  });

  // (Local ONNX models removed)

  // File upload endpoint
  app.post("/api/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      let text = "";
      const fileType = req.file.originalname.toLowerCase();

      if (fileType.endsWith(".txt")) {
        text = decodeTextBuffer(req.file.buffer);
      } else if (fileType.endsWith(".epub")) {
        if (req.file.buffer[0] !== 0x50 || req.file.buffer[1] !== 0x4b) {
          return res.status(400).json({ error: "The uploaded EPUB is not a valid ZIP-based EPUB file" });
        }
        text = await parseEpub(req.file.buffer);
      } else {
        return res.status(400).json({ error: "Unsupported file type" });
      }

      if (text.length > MAX_EXTRACTED_TEXT_CHARS || Buffer.byteLength(text, "utf8") > MAX_EXTRACTED_TEXT_BYTES) {
        return res.status(413).json({ error: "Extracted text exceeds the 5 MB processing limit" });
      }

      // Calculate stats
      const wordCount = countWords(text);
      const charCount = text.length;

      res.json({
        fileName: req.file.originalname,
        fileType: req.file.mimetype,
        text,
        wordCount,
        charCount,
      });
    } catch (error) {
      console.error("Upload error:", error);
      res.status(400).json({
        error: error instanceof Error ? error.message : "Failed to process file",
      });
    }
  });

  // PDF OCR endpoints
  app.get("/api/pdf-ocr/status", (_req, res) => {
    res.json(pdfOcrService.getStatus());
  });

  app.post("/api/pdf-ocr/config", (req, res) => {
    try {
      const body = (req.body ?? {}) as Partial<{
        pythonPath: string;
        deepseekRepoPath: string;
        huggingFaceRepoId: string;
        huggingFaceRevision: string;
      }>;
      pdfOcrService.updateConfig(body);
      res.json({ config: pdfOcrService.getConfig() });
    } catch (error) {
      console.error("Failed to update PDF OCR config:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to update config",
      });
    }
  });

  app.post("/api/pdf-ocr/download-models", (req, res) => {
    try {
      const task = pdfOcrService.downloadModels();
      task.catch((error) => {
        console.error("PDF OCR model download failed:", error);
      });
      res.json(pdfOcrService.getStatus());
    } catch (error) {
      res.status(409).json({
        error: error instanceof Error ? error.message : "Model download already in progress",
      });
    }
  });

  app.post("/api/pdf-ocr/process", pdfUpload.single("pdf"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "PDF file is required" });
      }
      if (req.file.buffer.subarray(0, 1024).indexOf(Buffer.from("%PDF-")) < 0) {
        return res.status(400).json({ error: "The uploaded file is not a valid PDF" });
      }

      const totalPages = estimatePdfPageCount(req.file.buffer);
      const job = await pdfOcrService.startJob({
        pdfBuffer: req.file.buffer,
        fileName: req.file.originalname,
        totalPages,
      });

      res.json({ job, totalPages });
    } catch (error) {
      console.error("PDF OCR start error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to start PDF OCR",
      });
    }
  });

  app.get("/api/pdf-ocr/jobs/:id", (req, res) => {
    const job = pdfOcrService.getJob(req.params.id);
    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }
    res.json({ job });
  });

  app.post("/api/pdf-ocr/jobs/:id/cancel", (req, res) => {
    const job = pdfOcrService.cancelJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });
    if (job.status !== "cancelled") {
      return res.status(409).json({ error: `Job is already ${job.status}`, job });
    }
    res.json({ job });
  });

  app.get("/api/pdf-ocr/jobs/:id/text", async (req, res) => {
    try {
      const outputPath = pdfOcrService.getJobOutputPath(req.params.id);
      if (!outputPath) {
        return res.status(404).json({ error: "Text not available" });
      }
      if (!fs.existsSync(outputPath)) {
        return res.status(404).json({ error: "Text file missing" });
      }
      const text = await fs.promises.readFile(outputPath, "utf-8");
      res.type("text/plain").send(text);
    } catch (error) {
      console.error("Failed to read PDF OCR output:", error);
      res.status(500).json({ error: "Failed to read OCR output" });
    }
  });

  // IndexTTS control endpoints
  app.get("/api/tts/status", (_req, res) => {
    res.json(indexTtsService.getStatus());
  });

  app.post("/api/tts/download", (_req, res) => {
    const status = indexTtsService.getStatus();
    if (status.modelsReady) {
      return res.json(status);
    }
    if (status.downloadStatus === "in-progress") {
      return res.status(409).json({ error: "Download already in progress" });
    }
    indexTtsService
      .downloadModels()
      .catch((error) => console.error("IndexTTS download failed:", error));
    res.json(indexTtsService.getStatus());
  });

  app.post("/api/tts/load", (_req, res) => {
    const status = indexTtsService.getStatus();
    if (!status.modelsReady) {
      return res.status(409).json({ error: "Download models before loading" });
    }
    if (status.loadStatus === "in-progress") {
      return res.status(409).json({ error: "Load already in progress" });
    }
    indexTtsService.loadModels().catch((error) => console.error("IndexTTS load failed:", error));
    res.json(indexTtsService.getStatus());
  });

  app.post(
    "/api/tts/synthesize",
    ttsUpload.fields([
      { name: "voice", maxCount: 1 },
      { name: "script", maxCount: 1 },
    ]),
    async (req, res) => {
      try {
        const files = (req.files || {}) as Record<string, Express.Multer.File[]>;
        const voiceFile = await resolveVoiceInput(files.voice?.[0], req.body?.voiceId);
        const scriptFile = files.script?.[0];

        if (!voiceFile) {
          return res.status(400).json({ error: "Voice prompt is required" });
        }

        let textContent = typeof req.body?.text === "string" ? req.body.text : "";
        let textFileName = scriptFile?.originalname ?? "script.txt";
        if (scriptFile) {
          textContent = decodeTextBuffer(scriptFile.buffer);
        }

        if (!textContent || textContent.trim().length === 0) {
          return res.status(400).json({ error: "Text input is required" });
        }
        assertSynthesisTextSize(textContent);

        const job = await indexTtsService.startSynthesis({
          voiceBuffer: voiceFile.buffer,
          voiceFileName: voiceFile.originalname,
          textContent,
          textFileName,
        });

        res.json({ job });
      } catch (error) {
        console.error("TTS synthesis error:", error);
        res.status(error instanceof RequestInputError ? 400 : 500).json({
          error: error instanceof Error ? error.message : "Failed to start synthesis",
        });
      }
    }
  );

  app.get("/api/tts/jobs/:id", (req, res) => {
    const job = indexTtsService.getJob(req.params.id);
    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }
    res.json({ job });
  });

  app.post("/api/tts/jobs/:id/cancel", (req, res) => {
    const job = indexTtsService.cancelJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });
    if (job.status !== "cancelled") {
      return res.status(409).json({ error: `Job is already ${job.status}`, job });
    }
    res.json({ job });
  });

  app.get("/api/tts/jobs/:id/audio", (req, res) => {
    const job = indexTtsService.getJob(req.params.id);
    if (!job || job.status !== "completed" || !job.outputFile) {
      return res.status(404).json({ error: "Audio not ready" });
    }
    const filePath = path.resolve(job.outputFile);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "Audio file missing" });
    }
    res.type("audio/wav");
    const disposition = req.query.download === "1" ? "attachment" : "inline";
    res.setHeader("Content-Disposition", `${disposition}; filename="tts-${job.id}.wav"`);
    res.sendFile(filePath);
  });

  // VibeVoice control endpoints
  app.get("/api/vibevoice/status", (_req, res) => {
    res.json(vibevoiceService.getStatus());
  });

  app.post("/api/vibevoice/setup", (req, res) => {
    const status = vibevoiceService.getStatus();
    if (status.setupStatus === "in-progress") {
      return res.status(409).json({ error: "Setup already in progress" });
    }
    vibevoiceService
      .startSetup()
      .catch((error) => console.error("VibeVoice setup failed:", error));
    res.json(vibevoiceService.getStatus());
  });

  app.post(
    "/api/vibevoice/synthesize",
    vibevoiceUpload.fields([
      { name: "voice1", maxCount: 1 },
      { name: "voice2", maxCount: 1 },
      { name: "voice3", maxCount: 1 },
      { name: "voice4", maxCount: 1 },
      { name: "script", maxCount: 1 },
    ]),
    async (req, res) => {
      try {
        const files = (req.files || {}) as Record<string, Express.Multer.File[]>;
        const voiceSlots = await Promise.all([
          resolveVoiceInput(files.voice1?.[0], req.body?.voiceId1),
          resolveVoiceInput(files.voice2?.[0], req.body?.voiceId2),
          resolveVoiceInput(files.voice3?.[0], req.body?.voiceId3),
          resolveVoiceInput(files.voice4?.[0], req.body?.voiceId4),
        ]);
        const lastVoiceSlot = voiceSlots.reduce(
          (last, file, index) => (file ? index : last),
          -1
        );
        if (lastVoiceSlot < 0) {
          return res.status(400).json({ error: "At least one voice reference is required" });
        }
        if (voiceSlots.slice(0, lastVoiceSlot + 1).some((file) => !file)) {
          return res.status(400).json({
            error: "Fill voice references in order without empty slots (Voice 1, then Voice 2, and so on)",
          });
        }
        const voiceFiles = voiceSlots.slice(0, lastVoiceSlot + 1) as ResolvedVoiceInput[];
        const scriptFile = files.script?.[0];

        let textContent = typeof req.body?.text === "string" ? req.body.text : "";
        let textFileName = scriptFile?.originalname ?? "script.txt";
        if (scriptFile) {
          textContent = decodeTextBuffer(scriptFile.buffer);
        }

        if (!textContent || textContent.trim().length === 0) {
          return res.status(400).json({ error: "Text input is required" });
        }
        assertSynthesisTextSize(textContent);

        const rawGuidanceScale = req.body?.guidanceScale;
        const parsedGuidanceScale =
          typeof rawGuidanceScale === "string" && rawGuidanceScale.trim().length > 0
            ? Number(rawGuidanceScale)
            : undefined;
        const guidanceScale = Number.isFinite(parsedGuidanceScale)
          ? Math.min(3, Math.max(0.5, parsedGuidanceScale as number))
          : undefined;
        const style =
          typeof req.body?.style === "string" && req.body.style.trim().length > 0
            ? req.body.style.trim()
            : undefined;
        if (style && style.length > 500) {
          return res.status(400).json({ error: "Style guidance must be 500 characters or fewer" });
        }

        const modelId =
          typeof req.body?.modelId === "string" && req.body.modelId.trim().length > 0
            ? req.body.modelId.trim()
            : undefined;
        if (
          modelId &&
          !vibevoiceService.getStatus().availableModels.some((model) => model.id === modelId)
        ) {
          return res.status(400).json({ error: "Choose a VibeVoice model that has been installed locally" });
        }

        const job = await vibevoiceService.startSynthesis({
          voiceBuffers: voiceFiles.map((f) => f.buffer),
          voiceFileNames: voiceFiles.map((f) => f.originalname),
          voiceBuffer: undefined,
          voiceFileName: voiceFiles[0]?.originalname,
          textContent,
          textFileName,
          style,
          guidanceScale,
          modelId,
        });

        res.json({ job });
      } catch (error) {
        console.error("VibeVoice synthesis error:", error);
        res.status(error instanceof RequestInputError ? 400 : 500).json({
          error: error instanceof Error ? error.message : "Failed to start synthesis",
        });
      }
    }
  );

  app.get("/api/vibevoice/jobs/:id", (req, res) => {
    const job = vibevoiceService.getJob(req.params.id);
    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }
    res.json({ job });
  });

  app.post("/api/vibevoice/jobs/:id/cancel", (req, res) => {
    const job = vibevoiceService.cancelJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });
    if (job.status !== "cancelled") {
      return res.status(409).json({ error: `Job is already ${job.status}`, job });
    }
    res.json({ job });
  });

  app.get("/api/vibevoice/jobs/:id/audio", (req, res) => {
    const job = vibevoiceService.getJob(req.params.id);
    if (!job || job.status !== "completed" || !job.outputFile) {
      return res.status(404).json({ error: "Audio not ready" });
    }
    const filePath = path.resolve(job.outputFile);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "Audio file missing" });
    }
    res.type("audio/wav");
    const disposition = req.query.download === "1" ? "attachment" : "inline";
    res.setHeader("Content-Disposition", `${disposition}; filename="vibevoice-${job.id}.wav"`);
    res.sendFile(filePath);
  });

  // Qwen3-TTS and MOSS-TTS share a target-aware speech API. Hosted requests
  // call the allowlisted official Gradio Spaces with the server-held HF token.
  app.get("/api/speech/status", (_req, res) => {
    res.json(speechService.getStatus());
  });

  app.post("/api/speech/:engine/setup", (req, res) => {
    const engineResult = speechEngineSchema.safeParse(req.params.engine);
    if (!engineResult.success) return res.status(404).json({ error: "Unknown speech engine" });
    const current = speechService
      .getStatus()
      .engines.find((item) => item.engine === engineResult.data);
    if (current?.setupStatus === "in-progress") {
      return res.status(409).json({ error: "Setup is already in progress" });
    }
    if (!current?.runtimeConfigured) {
      return res.status(409).json({
        error: `Run VoiceForge.cmd setup-${engineResult.data}, restart VoiceForge, then download the pinned model.`,
      });
    }
    const modelId = typeof req.body?.modelId === "string" ? req.body.modelId.trim() : undefined;
    try {
      speechService.validateSetupRequest(engineResult.data, modelId);
    } catch (error) {
      return res.status(409).json({ error: error instanceof Error ? error.message : "Setup cannot start" });
    }
    speechService
      .startSetup(engineResult.data, modelId)
      .catch((error) => console.error(`${engineResult.data} setup failed:`, error));
    res.status(202).json(speechService.getStatus());
  });

  app.post("/api/speech/:engine/setup/cancel", async (req, res) => {
    const engineResult = speechEngineSchema.safeParse(req.params.engine);
    if (!engineResult.success) return res.status(404).json({ error: "Unknown speech engine" });
    try {
      await speechService.cancelSetup(engineResult.data);
      res.json(speechService.getStatus());
    } catch (error) {
      res.status(409).json({ error: error instanceof Error ? error.message : "Setup could not be stopped" });
    }
  });

  app.post(
    "/api/speech/synthesize",
    speechUpload.fields([
      { name: "voice", maxCount: 1 },
      { name: "script", maxCount: 1 },
    ]),
    async (req, res) => {
      try {
        const engineResult = speechEngineSchema.safeParse(req.body?.engine);
        const targetResult = speechExecutionTargetSchema.safeParse(req.body?.target);
        if (!engineResult.success || !targetResult.success) {
          return res.status(400).json({ error: "Choose a valid speech engine and execution target" });
        }
        const engine = engineResult.data;
        const target = targetResult.data;
        const files = (req.files || {}) as Record<string, Express.Multer.File[]>;
        const voiceFile = await resolveVoiceInput(files.voice?.[0], req.body?.voiceId);
        const scriptFile = files.script?.[0];

        let text = typeof req.body?.text === "string" ? req.body.text : "";
        if (scriptFile) text = decodeTextBuffer(scriptFile.buffer);
        text = text.trim();
        if (!text) return res.status(400).json({ error: "Text input is required" });

        const remoteLimit = engine === "qwen" ? 1_200 : 5_000;
        const limit = target === "hf-space" ? remoteLimit : MAX_SYNTHESIS_TEXT_CHARS;
        if (text.length > limit) {
          return res.status(400).json({
            error:
              target === "hf-space"
                ? `The official ${engine === "qwen" ? "Qwen" : "MOSS"} Space is limited to ${limit.toLocaleString()} characters per VoiceForge request. Use Local for long-form synthesis.`
                : `Text input exceeds the ${limit.toLocaleString()} character limit`,
          });
        }

        const mode =
          typeof req.body?.mode === "string" && req.body.mode.trim()
            ? req.body.mode.trim()
            : engine === "qwen"
              ? "clone"
              : "direct";
        const runtimeStatus = speechService
          .getStatus()
          .engines.find((item) => item.engine === engine);
        const allowedModes = target === "local" ? runtimeStatus?.localModes : runtimeStatus?.hostedModes;
        if (!allowedModes?.includes(mode)) {
          return res.status(400).json({ error: `${mode} mode is not supported for the selected target` });
        }
        const requiresVoice =
          (engine === "qwen" && mode === "clone") ||
          (engine === "moss" && mode !== "direct");
        if (requiresVoice && !voiceFile) {
          return res.status(400).json({ error: "This synthesis mode requires a voice reference" });
        }

        const finiteNumber = (
          value: unknown,
          fallback: number,
          minimum: number,
          maximum: number
        ) => {
          const parsed = typeof value === "string" && value.trim() ? Number(value) : Number.NaN;
          return Number.isFinite(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
        };
        const booleanValue = (value: unknown) =>
          value === true || (typeof value === "string" && ["1", "true", "yes", "on"].includes(value.toLowerCase()));

        const referenceText =
          typeof req.body?.referenceText === "string" && req.body.referenceText.trim()
            ? req.body.referenceText.trim().slice(0, 20_000)
            : voiceFile?.transcript?.trim().slice(0, 20_000) || undefined;
        const language =
          typeof req.body?.language === "string" && req.body.language.trim().length <= 40
            ? req.body.language.trim()
            : undefined;
        const modelId =
          typeof req.body?.modelId === "string" && req.body.modelId.trim().length <= 120
            ? req.body.modelId.trim()
            : undefined;
        const selectedModel = modelId || (engine === "qwen" ? "Qwen/Qwen3-TTS-12Hz-0.6B-Base" : "OpenMOSS-Team/MOSS-TTS-v1.5");
        const allowedModels = engine === "qwen"
          ? new Set(["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"])
          : new Set(["OpenMOSS-Team/MOSS-TTS-v1.5"]);
        if (!allowedModels.has(selectedModel)) {
          return res.status(400).json({ error: "Choose a supported pinned model" });
        }
        if (target === "local" && !runtimeStatus?.runtimeConfigured) {
          return res.status(409).json({ error: `Run VoiceForge.cmd setup-${engine} and restart VoiceForge first` });
        }
        if (target === "local" && !runtimeStatus?.availableModels.includes(selectedModel)) {
          return res.status(409).json({ error: "Download the selected pinned model before local synthesis" });
        }
        if (target === "hf-space" && !speechService.getStatus().tokenConfigured) {
          return res.status(409).json({ error: "Add a Hugging Face token before using ZeroGPU" });
        }
        const modelSize = ["0.6B", "1.7B"].includes(req.body?.modelSize)
          ? req.body.modelSize
          : undefined;
        const speaker = [
          "Aiden",
          "Dylan",
          "Eric",
          "Ono_anna",
          "Ryan",
          "Serena",
          "Sohee",
          "Uncle_fu",
          "Vivian",
        ].includes(req.body?.speaker)
          ? req.body.speaker
          : undefined;

        const job = await speechService.startSynthesis({
          engine,
          target,
          mode,
          text,
          voiceBuffer: voiceFile?.buffer,
          voiceFileName: voiceFile?.originalname,
          modelId: selectedModel,
          referenceText,
          language,
          xVectorOnly: booleanValue(req.body?.xVectorOnly),
          voiceDescription:
            typeof req.body?.voiceDescription === "string"
              ? req.body.voiceDescription.trim().slice(0, 1_000)
              : undefined,
          speaker,
          instruction:
            typeof req.body?.instruction === "string"
              ? req.body.instruction.trim().slice(0, 1_000)
              : undefined,
          modelSize,
          durationControl: booleanValue(req.body?.durationControl),
          durationTokens: Math.round(finiteNumber(req.body?.durationTokens, 1, 1, 4_096)),
          temperature: finiteNumber(req.body?.temperature, 1.7, 0.1, 3),
          topP: finiteNumber(req.body?.topP, 0.8, 0.1, 1),
          topK: Math.round(finiteNumber(req.body?.topK, 25, 1, 200)),
          repetitionPenalty: finiteNumber(req.body?.repetitionPenalty, 1, 0.5, 2),
          maxNewTokens: Math.round(finiteNumber(req.body?.maxNewTokens, 4_096, 128, 8_192)),
          maxChars: Math.round(finiteNumber(req.body?.maxChars, engine === "qwen" ? 320 : 1_800, 50, 10_000)),
          gapMs: Math.round(finiteNumber(req.body?.gapMs, 120, 0, 2_000)),
        });
        res.status(202).json({ job });
      } catch (error) {
        console.error("Speech synthesis error:", error);
        res.status(error instanceof RequestInputError ? 400 : 500).json({
          error: error instanceof Error ? error.message : "Failed to start speech synthesis",
        });
      }
    }
  );

  app.get("/api/speech/jobs/:id", (req, res) => {
    const job = speechService.getJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });
    res.json({ job });
  });

  app.post("/api/speech/jobs/:id/cancel", (req, res) => {
    const job = speechService.cancelJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });
    if (job.status !== "cancelled") {
      return res.status(409).json({ error: `Job is already ${job.status}`, job });
    }
    res.json({ job });
  });

  app.get("/api/speech/jobs/:id/audio", (req, res) => {
    const job = speechService.getJob(req.params.id);
    const outputPath = speechService.getJobOutputPath(req.params.id);
    if (!job || !outputPath || !fs.existsSync(outputPath)) {
      return res.status(404).json({ error: "Audio not ready" });
    }
    res.type("audio/wav");
    const disposition = req.query.download === "1" ? "attachment" : "inline";
    res.setHeader(
      "Content-Disposition",
      `${disposition}; filename="${job.engine}-${job.id}.wav"`
    );
    res.sendFile(outputPath);
  });

  const pdfOcrWss = new WebSocketServer({ noServer: true, maxPayload: 64 * 1024 });

  pdfOcrWss.on("connection", (ws: WebSocket) => {
    const unsubscribe = pdfOcrService.subscribe((message) => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(message));
        } catch (error) {
          console.error("PDF OCR WebSocket send error:", error);
        }
      }
    });

    ws.on("close", () => {
      unsubscribe();
    });

    ws.on("error", (error) => {
      console.error("PDF OCR WebSocket error:", error);
    });
  });

  // WebSocket server for IndexTTS updates
  const ttsWss = new WebSocketServer({ noServer: true, maxPayload: 64 * 1024 });

  ttsWss.on("connection", (ws: WebSocket) => {
    const unsubscribe = indexTtsService.subscribe((message) => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(message));
        } catch (error) {
          console.error("IndexTTS WebSocket send error:", error);
        }
      }
    });

    ws.on("close", () => {
      unsubscribe();
    });

    ws.on("error", (error) => {
      console.error("IndexTTS WebSocket error:", error);
    });
  });

  const vibevoiceWss = new WebSocketServer({ noServer: true, maxPayload: 64 * 1024 });

  vibevoiceWss.on("connection", (ws: WebSocket) => {
    const unsubscribe = vibevoiceService.subscribe((message) => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(message));
        } catch (error) {
          console.error("VibeVoice WebSocket send error:", error);
        }
      }
    });

    ws.on("close", () => {
      unsubscribe();
    });

    ws.on("error", (error) => {
      console.error("VibeVoice WebSocket error:", error);
    });
  });

  const speechWss = new WebSocketServer({ noServer: true, maxPayload: 64 * 1024 });

  speechWss.on("connection", (ws: WebSocket) => {
    const unsubscribe = speechService.subscribe((message) => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(message));
        } catch (error) {
          console.error("Speech WebSocket send error:", error);
        }
      }
    });
    ws.on("close", unsubscribe);
    ws.on("error", (error) => console.error("Speech WebSocket error:", error));
  });

  // WebSocket server for real-time processing
  const wss = new WebSocketServer({ noServer: true, maxPayload: 6 * 1024 * 1024 });

  wss.on("connection", (ws: WebSocket) => {
    let activeController: AbortController | undefined;
    let processing = false;

    ws.on("message", async (message: Buffer) => {
      if (processing) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "error", payload: { message: "A processing job is already running on this connection" } } as WSMessage));
        }
        return;
      }

      try {
        const parsed = processTextRequestSchema.safeParse(JSON.parse(message.toString()));
        if (!parsed.success) {
          ws.send(
            JSON.stringify({
              type: "error",
              payload: {
                message: "Invalid processing request",
              },
            } as WSMessage)
          );
          return;
        }
        const { text, config } = parsed.data;
        processing = true;
        const controller = new AbortController();
        activeController = controller;

        // Send log entry
        const startLog: LogEntry = {
          id: nanoid(),
          timestamp: new Date(),
          type: "info",
          message: `Starting text processing with ${config.modelName}`,
          details: `Batch size: ${config.batchSize} sentences`,
        };

        ws.send(
          JSON.stringify({
            type: "log",
            payload: startLog,
          } as WSMessage)
        );

        // Split text to get total chunks
        const totalChunks = chunkTextBySentences(text, config.batchSize).length;

        // Process text with progress updates
        const processingResult = await textProcessor.processText(
          text,
          config,
          (progress) => {
            if (ws.readyState !== WebSocket.OPEN) return;
            // Send progress update
            ws.send(
              JSON.stringify({
                type: "progress",
                payload: {
                  progress: ((progress.chunkIndex + 1) / totalChunks) * 100,
                  currentChunk: progress.chunkIndex + 1,
                  totalChunks,
                  lastChunkMs: progress.lastChunkMs,
                  avgChunkMs: progress.avgChunkMs,
                  etaMs: progress.etaMs,
                  // optional usage metrics
                  inputTokens: progress.inputTokens,
                  outputTokens: progress.outputTokens,
                  inputCost: progress.inputCost,
                  outputCost: progress.outputCost,
                  totalInputTokens: progress.totalInputTokens,
                  totalOutputTokens: progress.totalOutputTokens,
                  totalCost: progress.totalCost,
                },
              } as WSMessage)
            );

            // Send chunk result
            ws.send(
              JSON.stringify({
                type: "chunk",
                payload: progress,
              } as WSMessage)
            );

            // Send log for retries and failures
            if (progress.status === "retry") {
              const retryLog: LogEntry = {
                id: nanoid(),
                timestamp: new Date(),
                type: "warning",
                message: `Chunk ${progress.chunkIndex + 1} validation failed, retrying...`,
                details: `Retry attempt ${progress.retryCount}`,
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: retryLog,
                } as WSMessage)
              );
            } else if (progress.status === "failed") {
              const failLog: LogEntry = {
                id: nanoid(),
                timestamp: new Date(),
                type: "error",
                message: `Chunk ${progress.chunkIndex + 1} failed after retries`,
                details: "Using original text for this chunk",
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: failLog,
                } as WSMessage)
              );
            } else if (progress.status === "success") {
              const successLog: LogEntry = {
                id: nanoid(),
                timestamp: new Date(),
                type: "success",
                message: `Chunk ${progress.chunkIndex + 1} processed successfully`,
                details: (typeof progress.inputTokens === 'number')
                  ? `Estimated tokens in/out: ${progress.inputTokens}/${progress.outputTokens} — cost: $${((progress.inputCost||0)+(progress.outputCost||0)).toFixed(4)} (total: $${(progress.totalCost||0).toFixed(4)})`
                  : undefined,
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: successLog,
                } as WSMessage)
              );
            }
          },
          () => ws.readyState !== WebSocket.OPEN || controller.signal.aborted,
          controller.signal
        );

        // Send completion
        if (ws.readyState !== WebSocket.OPEN) return;
        ws.send(
          JSON.stringify({
            type: "complete",
            payload: {
              processedText: processingResult.text,
              totalChunks: processingResult.totalChunks,
              failedChunks: processingResult.failedChunkIndexes.length,
              failedChunkIndexes: processingResult.failedChunkIndexes,
              totalInputTokens: processingResult.totalInputTokens,
              totalOutputTokens: processingResult.totalOutputTokens,
              totalCost: processingResult.totalCost,
            },
          } as WSMessage)
        );
        // Send cost summary log if available
        ws.send(
          JSON.stringify({
            type: "log",
            payload: {
              id: nanoid(),
              timestamp: new Date(),
              type: "info",
              message: processingResult.failedChunkIndexes.length > 0
                ? "Processing complete with warnings"
                : "Processing complete",
              details: processingResult.failedChunkIndexes.length > 0
                ? `${processingResult.failedChunkIndexes.length} chunk(s) used their original text after retry failures.`
                : "All chunks passed validation.",
            } as LogEntry,
          } as WSMessage)
        );
      } catch (error) {
        if (error instanceof ProcessingCancelledError || ws.readyState !== WebSocket.OPEN) {
          return;
        }
        console.error("Processing error:", error);
        ws.send(
          JSON.stringify({
            type: "error",
            payload: {
              message: error instanceof Error ? error.message : "Processing failed",
              details: error instanceof Error ? error.stack : undefined,
            },
          } as WSMessage)
        );
      } finally {
        processing = false;
        activeController = undefined;
      }
    });

    ws.on("close", () => {
      activeController?.abort();
    });

    ws.on("error", (error) => {
      activeController?.abort();
      console.error("WebSocket error:", error);
    });
  });

  // Handle WebSocket upgrade
  httpServer.on("upgrade", (request, socket, head) => {
    const bindHost = process.env.HOST || "127.0.0.1";
    const bindPort = Number.parseInt(process.env.PORT || "5000", 10);
    const policy = evaluateLocalRequest(bindHost, bindPort, request.headers.host, request.headers.origin);
    if (policy !== "allowed") {
      socket.write("HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
      socket.destroy();
      return;
    }
    if (!request.url) {
      socket.destroy();
      return;
    }

    if (request.url === "/ws/process") {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit("connection", ws, request);
      });
    } else if (request.url === "/ws/tts") {
      ttsWss.handleUpgrade(request, socket, head, (ws) => {
        ttsWss.emit("connection", ws, request);
      });
    } else if (request.url === "/ws/pdf-ocr") {
      pdfOcrWss.handleUpgrade(request, socket, head, (ws) => {
        pdfOcrWss.emit("connection", ws, request);
      });
    } else if (request.url === "/ws/vibevoice") {
      vibevoiceWss.handleUpgrade(request, socket, head, (ws) => {
        vibevoiceWss.emit("connection", ws, request);
      });
    } else if (request.url === "/ws/speech") {
      speechWss.handleUpgrade(request, socket, head, (ws) => {
        speechWss.emit("connection", ws, request);
      });
    } else if (process.env.NODE_ENV !== "development") {
      socket.destroy();
    }
  });

  return httpServer;
}
