import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { WebSocketServer, WebSocket } from "ws";
import AdmZip from "adm-zip";
import { parse } from "node-html-parser";
import { textProcessor } from "./text-processor";
import { llmService, setHuggingFaceApiToken, getHuggingFaceApiToken } from "./llm-service";
import { nanoid } from "nanoid";
import type { ProcessingConfig, LogEntry, WSMessage, HuggingFaceTokenStatus } from "@shared/schema";
import { huggingFaceTokenUpdateSchema, deterministicCleanRequestSchema } from "@shared/schema";
import { clampCharacterSampleSize, getCharacterSampleCeiling } from "@shared/model-utils";
import { indexTtsService } from "./tts-service";
import { vibevoiceService } from "./vibevoice-service";
import fs from "fs";
import path from "path";
import { applyDeterministicCleaning } from "./text-cleaner";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
});

const ttsUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 120 * 1024 * 1024, // 120MB limit for audio/text assets
  },
});

const vibevoiceUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 120 * 1024 * 1024,
  },
});

async function parseEpub(buffer: Buffer): Promise<string> {
  try {
    const zip = new AdmZip(buffer);
    const zipEntries = zip.getEntries();
    
    const textParts: string[] = [];
    
    // Find and parse all HTML/XHTML files in the EPUB
    for (const entry of zipEntries) {
      const fileName = entry.entryName.toLowerCase();
      
      // Look for content files (HTML/XHTML in OEBPS or similar directories)
      if ((fileName.endsWith('.html') || fileName.endsWith('.xhtml') || fileName.endsWith('.htm')) &&
          !fileName.includes('nav.') && !fileName.includes('toc.')) {
        
        try {
          const content = entry.getData().toString('utf-8');
          const root = parse(content);
          
          // Extract text from body
          const body = root.querySelector('body');
          if (body) {
            const text = body.textContent
              .replace(/\s+/g, ' ') // Normalize whitespace
              .trim();
            
            if (text.length > 0) {
              textParts.push(text);
            }
          }
        } catch (err) {
          console.error(`Error parsing entry ${entry.entryName}:`, err);
        }
      }
    }
    
    if (textParts.length === 0) {
      throw new Error("No readable content found in EPUB file");
    }
    
    return textParts.join('\n\n');
  } catch (error) {
    console.error("EPUB parsing error:", error);
    throw new Error("Failed to parse EPUB file: " + (error instanceof Error ? error.message : "Unknown error"));
  }
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

async function upsertEnvValue(filePath: string, key: string, value: string | undefined): Promise<void> {
  const normalizedValue = value?.trim();
  const exists = fs.existsSync(filePath);
  if (!exists && (!normalizedValue || normalizedValue.length === 0)) {
    return;
  }

  const raw = exists ? await fs.promises.readFile(filePath, "utf-8") : "";
  const lines = raw.length > 0 ? raw.split(/\r?\n/) : [];
  let updated = false;
  const output: string[] = [];

  for (const line of lines) {
    if (line === undefined) continue;
    const match = line.match(/^\s*(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (match && match[2] === key) {
      updated = true;
      if (normalizedValue && normalizedValue.length > 0) {
        const prefix = match[1] ?? "";
        output.push(`${prefix}${key}=${normalizedValue}`);
      }
    } else {
      output.push(line);
    }
  }

  if (!updated && normalizedValue && normalizedValue.length > 0) {
    output.push(`${key}=${normalizedValue}`);
  }

  while (output.length > 0 && output[output.length - 1].trim() === "") {
    output.pop();
  }

  const content = output.join("\n");
  await fs.promises.writeFile(filePath, content ? `${content}\n` : "");
}

async function persistHuggingFaceToken(token: string | undefined): Promise<void> {
  const envPath = path.resolve(process.cwd(), ".env");
  const envTxtPath = path.resolve(process.cwd(), "env.txt");
  await Promise.all([
    upsertEnvValue(envPath, "HUGGINGFACE_API_TOKEN", token),
    upsertEnvValue(envPath, "HF_TOKEN", token),
    upsertEnvValue(envTxtPath, "HUGGINGFACE_API_TOKEN", token),
    upsertEnvValue(envTxtPath, "HF_TOKEN", token),
  ]);
}

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);

  app.get("/api/settings/huggingface-token", (_req, res) => {
    res.json(buildTokenStatus());
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
      setHuggingFaceApiToken(nextToken);
      await persistHuggingFaceToken(nextToken);
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
        (config as any).concisePrompts !== false,
        (config as any).extendedExamples === true
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
    try {
      const { text, config } = req.body as {
        text: string;
        config: ProcessingConfig;
      };

      if (!text || !config) {
        return res.status(400).json({ error: "Missing text or configuration" });
      }

      // Process just the first chunk (up to batchSize sentences)
      const sentences = text.match(/[^.!?]+(?:[.!?]+|$)/g) || [text];
      const testChunk = sentences.slice(0, config.batchSize).join(" ");

      const result = await llmService.processChunk({
        text: testChunk,
        cleaningOptions: config.cleaningOptions,
        speakerConfig: config.speakerConfig,
        modelSource: config.modelSource,
        modelName: config.modelName,
        ollamaModelName: (config as any).ollamaModelName,
        customInstructions: config.customInstructions,
        extendedExamples: (config as any).extendedExamples === true,
      });

      res.json({
        originalChunk: testChunk,
        processedChunk: result.text,
        sentenceCount: Math.min(sentences.length, config.batchSize),
        usage: result.usage,
      });
    } catch (error) {
      console.error("Test chunk error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to test chunk",
      });
    }
  });

  // Extract character names endpoint
  app.post("/api/extract-characters", async (req, res) => {
    try {
      const { text, sampleSize, includeNarrator, modelSource, modelName, ollamaModelName } = req.body as {
        text: string;
        sampleSize: number;
        includeNarrator: boolean;
        modelSource?: string;
        modelName: string;
        ollamaModelName?: string;
      };

      if (!text || !sampleSize) {
        return res.status(400).json({ error: "Missing text or sample size" });
      }

      const resolvedSource = ((modelSource as any) || 'api') as 'api' | 'ollama';
      const maxSample = getCharacterSampleCeiling(resolvedSource, ollamaModelName);
      const effectiveSampleSize = clampCharacterSampleSize(sampleSize, resolvedSource, ollamaModelName);

      // Extract sample from text
      const sentences = text.match(/[^.!?]+(?:[.!?]+|$)/g) || [text];
      const sampleText = sentences.slice(0, effectiveSampleSize).join(" ");

      // Use LLM to extract character names and narrator identity
      const { characters, narratorCharacterName } = await llmService.extractCharacters({
        text: sampleText,
        includeNarrator,
        modelSource: resolvedSource,
        modelName: modelName || "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ollamaModelName,
      });

      res.json({
        characters,
        narratorCharacterName,
        sampleSentenceCount: Math.min(sentences.length, effectiveSampleSize),
        sampleLimit: maxSample,
      });
    } catch (error) {
      console.error("Character extraction error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to extract characters",
      });
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
        text = req.file.buffer.toString("utf-8");
      } else if (fileType.endsWith(".epub")) {
        text = await parseEpub(req.file.buffer);
      } else {
        return res.status(400).json({ error: "Unsupported file type" });
      }

      // Calculate stats
      const wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;
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
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to process file",
      });
    }
  });

  // IndexTTS control endpoints
  app.get("/api/tts/status", (_req, res) => {
    res.json(indexTtsService.getStatus());
  });

  app.post("/api/tts/download", (req, res) => {
    const { repoId } = (req.body ?? {}) as { repoId?: string };
    const status = indexTtsService.getStatus();
    if (status.downloadStatus === "in-progress") {
      return res.status(409).json({ error: "Download already in progress" });
    }
    const trimmedRepo =
      typeof repoId === "string" && repoId.trim().length > 0 ? repoId.trim() : undefined;
    indexTtsService
      .downloadModels(trimmedRepo)
      .catch((error) => console.error("IndexTTS download failed:", error));
    res.json(indexTtsService.getStatus());
  });

  app.post("/api/tts/load", (_req, res) => {
    const status = indexTtsService.getStatus();
    if (status.downloadStatus !== "completed") {
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
        const voiceFile = files.voice?.[0];
        const scriptFile = files.script?.[0];

        if (!voiceFile) {
          return res.status(400).json({ error: "Voice prompt is required" });
        }

        let textContent = typeof req.body?.text === "string" ? req.body.text : "";
        let textFileName = scriptFile?.originalname ?? "script.txt";
        if (scriptFile) {
          textContent = scriptFile.buffer.toString("utf-8");
        }

        if (!textContent || textContent.trim().length === 0) {
          return res.status(400).json({ error: "Text input is required" });
        }

        const rawSteps = Number(req.body?.steps ?? 20);
        const steps = Number.isFinite(rawSteps) ? rawSteps : 20;

        const job = await indexTtsService.startSynthesis({
          voiceBuffer: voiceFile.buffer,
          voiceFileName: voiceFile.originalname,
          textContent,
          textFileName,
          steps,
        });

        res.json({ job });
      } catch (error) {
        console.error("TTS synthesis error:", error);
        res.status(500).json({
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
    res.setHeader("Content-Disposition", `attachment; filename="tts-${job.id}.wav"`);
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
    const { repoUrl, branch } = (req.body ?? {}) as { repoUrl?: string; branch?: string };
    vibevoiceService
      .startSetup(repoUrl, branch)
      .catch((error) => console.error("VibeVoice setup failed:", error));
    res.json(vibevoiceService.getStatus());
  });

  app.post(
    "/api/vibevoice/synthesize",
    vibevoiceUpload.fields([
      { name: "voice", maxCount: 1 },
      { name: "script", maxCount: 1 },
    ]),
    async (req, res) => {
      try {
        const files = (req.files || {}) as Record<string, Express.Multer.File[]>;
        const voiceFile = files.voice?.[0];
        const scriptFile = files.script?.[0];

        let textContent = typeof req.body?.text === "string" ? req.body.text : "";
        let textFileName = scriptFile?.originalname ?? "script.txt";
        if (scriptFile) {
          textContent = scriptFile.buffer.toString("utf-8");
        }

        if (!textContent || textContent.trim().length === 0) {
          return res.status(400).json({ error: "Text input is required" });
        }

        const rawTemperature = req.body?.temperature;
        const parsedTemperature =
          typeof rawTemperature === "string" && rawTemperature.trim().length > 0
            ? Number(rawTemperature)
            : undefined;
        const temperature = Number.isFinite(parsedTemperature) ? parsedTemperature : undefined;
        const style =
          typeof req.body?.style === "string" && req.body.style.trim().length > 0
            ? req.body.style.trim()
            : undefined;

        const job = await vibevoiceService.startSynthesis({
          voiceBuffer: voiceFile?.buffer,
          voiceFileName: voiceFile?.originalname,
          textContent,
          textFileName,
          style,
          temperature,
        });

        res.json({ job });
      } catch (error) {
        console.error("VibeVoice synthesis error:", error);
        res.status(500).json({
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
    res.setHeader("Content-Disposition", `attachment; filename="vibevoice-${job.id}.wav"`);
    res.sendFile(filePath);
  });

  // WebSocket server for IndexTTS updates
  const ttsWss = new WebSocketServer({ noServer: true });

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

  const vibevoiceWss = new WebSocketServer({ noServer: true });

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

  // WebSocket server for real-time processing
  const wss = new WebSocketServer({ noServer: true });

  wss.on("connection", (ws: WebSocket) => {
    ws.on("message", async (message: Buffer) => {
      try {
        const data = JSON.parse(message.toString());
        const { text, config } = data as {
          text: string;
          config: ProcessingConfig;
        };

        if (!text || !config) {
          ws.send(
            JSON.stringify({
              type: "error",
              payload: {
                message: "Missing text or configuration",
              },
            } as WSMessage)
          );
          return;
        }

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
        const sentences = text.match(/[^.!?]+(?:[.!?]+|$)/g) || [text];
        const totalChunks = Math.ceil(sentences.length / config.batchSize);

        // Process text with progress updates
        const processedText = await textProcessor.processText(
          text,
          config,
          (progress) => {
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
                  ? `Chunk tokens in/out: ${progress.inputTokens}/${progress.outputTokens} â€” cost: $${((progress.inputCost||0)+(progress.outputCost||0)).toFixed(4)} (total: $${(progress.totalCost||0).toFixed(4)})`
                  : undefined,
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: successLog,
                } as WSMessage)
              );
            }
          }
        );

        // Send completion
        ws.send(
          JSON.stringify({
            type: "complete",
            payload: {
              processedText,
              totalChunks,
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
              message: "Processing complete",
              details: "See progress logs for token and cost summary per chunk.",
            } as LogEntry,
          } as WSMessage)
        );
      } catch (error) {
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
      }
    });

    ws.on("error", (error) => {
      console.error("WebSocket error:", error);
    });
  });

  // Handle WebSocket upgrade
  httpServer.on("upgrade", (request, socket, head) => {
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
    } else if (request.url === "/ws/vibevoice") {
      vibevoiceWss.handleUpgrade(request, socket, head, (ws) => {
        vibevoiceWss.emit("connection", ws, request);
      });
    } else {
      socket.destroy();
    }
  });

  return httpServer;
}
