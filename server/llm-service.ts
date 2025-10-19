import { HfInference } from "@huggingface/inference";
import type { CleaningOptions, SpeakerConfig, ModelSource } from "@shared/schema";
import { buildOllamaOptions, isThinkingOllamaModel } from "@shared/model-utils";
import fs from "fs";
import path from "path";
import { applyDeterministicCleaning } from "./text-cleaner";

// Check if HuggingFace API token is available
let apiToken: string | undefined =
  (process.env.HUGGINGFACE_API_TOKEN || process.env.HF_TOKEN || "").trim() || undefined;
if (!apiToken) {
  console.warn("⚠️  HUGGINGFACE_API_TOKEN not found in environment variables!");
  console.warn("   API mode will not work. Please either:");
  console.warn("   1. Add HUGGINGFACE_API_TOKEN to your environment (dotenv/.env)");
  console.warn("   2. Use Ollama as the model source");
}

let hf = new HfInference(apiToken);

export function getHuggingFaceApiToken(): string | undefined {
  return apiToken;
}

export function setHuggingFaceApiToken(token: string | undefined) {
  const trimmed = token?.trim();
  apiToken = trimmed && trimmed.length > 0 ? trimmed : undefined;
  if (apiToken) {
    process.env.HUGGINGFACE_API_TOKEN = apiToken;
    process.env.HF_TOKEN = apiToken;
  } else {
    delete process.env.HUGGINGFACE_API_TOKEN;
    delete process.env.HF_TOKEN;
  }
  hf = new HfInference(apiToken);
}
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const HF_PROVIDER = (process.env.HF_PROVIDER || "hf-inference").trim();

// Debug logging helpers
const LLM_DEBUG = String(process.env.LLM_DEBUG || "").toLowerCase();
const DEBUG_ENABLED = LLM_DEBUG === "1" || LLM_DEBUG === "true" || LLM_DEBUG === "yes";

function redact(value: string): string {
  if (!value) return value;
  // Keep prefix e.g., "Bearer " but redact token content
  const bearerMatch = value.match(/^\s*Bearer\s+(.+)/i);
  if (bearerMatch) {
    const token = bearerMatch[1];
    const shown = token.slice(0, 6);
    return `Bearer ${shown}…[REDACTED]`;
  }
  // Generic token redaction
  if (value.length > 8) {
    return `${value.slice(0, 4)}…[REDACTED]`;
  }
  return "[REDACTED]";
}

function normalizeHeaders(headers?: HeadersInit): Record<string, string> {
  const out: Record<string, string> = {};
  if (!headers) return out;
  if (headers instanceof Headers) {
    headers.forEach((v, k) => (out[k] = v));
    return out;
  }
  if (Array.isArray(headers)) {
    for (const [k, v] of headers) out[k] = String(v);
    return out;
  }
  for (const [k, v] of Object.entries(headers)) out[k] = String(v);
  return out;
}

function sanitizeHeaders(headers?: HeadersInit): Record<string, string> {
  const h = normalizeHeaders(headers);
  const lowered: Record<string, string> = {};
  for (const [k, v] of Object.entries(h)) {
    const key = k.toLowerCase();
    if (["authorization", "x-api-key", "api-key", "proxy-authorization"].includes(key)) {
      lowered[k] = redact(v);
    } else {
      lowered[k] = v;
    }
  }
  return lowered;
}

function summarizeBody(body?: BodyInit | null): string | Record<string, unknown> | undefined {
  if (!body) return undefined;
  if (typeof body === "string") {
    try {
      const parsed = JSON.parse(body);
      // Avoid printing large inputs; truncate prompt-like fields
      const clone: Record<string, unknown> = { ...parsed };
      for (const key of ["prompt", "inputs", "text"]) {
        const val = clone[key];
        if (typeof val === "string" && val.length > 500) {
          clone[key] = `${val.slice(0, 500)}… [truncated ${val.length - 500} chars]`;
        }
      }
      return clone;
    } catch {
      return body.length > 600 ? `${body.slice(0, 600)}… [truncated ${body.length - 600} chars]` : body;
    }
  }
  if (body instanceof Blob) return { blob: true, size: (body as Blob).size };
  if (body instanceof ArrayBuffer) return { arrayBuffer: true, size: (body as ArrayBuffer).byteLength };
  return { body: "[unprintable type]" };
}

function logRequest(label: string, url: string | URL, init?: RequestInit) {
  if (!DEBUG_ENABLED) return;
  const method = init?.method || "GET";
  const headers = sanitizeHeaders(init?.headers);
  const body = summarizeBody(init?.body ?? undefined);
  console.debug(`[LLM DEBUG] ${label} request`, { method, url: String(url), headers, body });
}

function logResponse(label: string, res: Response, info?: { bodyPreview?: string | object }) {
  if (!DEBUG_ENABLED) return;
  const headers: Record<string, string> = {};
  res.headers.forEach((v, k) => (headers[k] = v));
  console.debug(`[LLM DEBUG] ${label} response`, {
    url: (res as any).url,
    ok: res.ok,
    status: res.status,
    statusText: res.statusText,
    headers,
    ...(info || {}),
  });
}

function createLoggingFetch(label: string): typeof fetch {
  const nativeFetch = globalThis.fetch.bind(globalThis);
  return async (url: any, init?: RequestInit): Promise<Response> => {
    logRequest(label, url, init);
    const res = await nativeFetch(url, init);
    try {
      const clone = res.clone();
      const ct = clone.headers.get('content-type') || '';
      if (ct.startsWith('application/json') || ct.startsWith('text/')) {
        const text = await clone.text();
        const bodyPreview = text.length > 1200 ? `${text.slice(0, 1200)}… [truncated ${text.length - 1200} chars]` : text;
        logResponse(label, res, { bodyPreview });
      } else {
        logResponse(label, res);
      }
    } catch {
      logResponse(label, res);
    }
    return res;
  };
}

function normalizeHuggingFaceModelId(model: string): string {
  // Fix common Llama 3.1 naming: meta-llama/Meta-Llama-3.1-* -> meta-llama/Llama-3.1-*
  if (/^meta-llama\/Meta-Llama-3\.1-/i.test(model)) {
    return model.replace(/meta-llama\/Meta-Llama-3\.1-/i, 'meta-llama/Llama-3.1-');
  }
  return model;
}

function parseProviderFromTaskError(message: string): string | undefined {
  const m = message.match(/Task 'text-generation' not supported for provider '([^']+)'/i);
  return m?.[1];
}

function supportsConversationalFromError(message: string): boolean {
  return /Available tasks:\s*conversational/i.test(message);
}

function getProviderAccessToken(provider: string): string | undefined {
  const mapping: Record<string, string[]> = {
    'fireworks-ai': [
      'FIREWORKS_API_KEY',
      'FIREWORKS_KEY',
      'PROVIDER_API_KEY',
      'HF_TOKEN', // allow user-provided env if they followed HF snippet
    ],
  };
  const vars = mapping[provider] || ['PROVIDER_API_KEY'];
  for (const name of vars) {
    const val = process.env[name];
    if (val && val.trim()) return val.trim();
  }
  return undefined;
}

export interface ProcessChunkOptions {
  text: string;
  cleaningOptions: CleaningOptions;
  speakerConfig?: SpeakerConfig;
  modelSource?: ModelSource;
  modelName: string; // API model name
  localModelName?: string; // Local model name
  ollamaModelName?: string; // Ollama model name
  customInstructions?: string;
  singlePass?: boolean;
  concisePrompts?: boolean;
  extendedExamples?: boolean;
}

type ModelPricing = { inCostPerM?: number; outCostPerM?: number };
type InferenceUsage = {
  inputTokens: number;
  outputTokens: number;
  inputCost: number;
  outputCost: number;
};

function estimateTokens(text: string): number {
  if (!text) return 0;
  const t = Math.ceil(text.trim().length / 4);
  return t > 0 ? t : 0;
}

function normalizeModelIdForLookup(id: string): string {
  return normalizeHuggingFaceModelId(id).toLowerCase();
}

let GOOD_MODELS_CACHE: Array<{ id: string; inCostPerM?: number; outCostPerM?: number; display?: string }> | null = null;
function loadGoodModels(): Array<{ id: string; inCostPerM?: number; outCostPerM?: number; display?: string }> {
  if (GOOD_MODELS_CACHE) return GOOD_MODELS_CACHE;
  try {
    const jsonPath = path.resolve(process.cwd(), 'good_models.json');
    if (fs.existsSync(jsonPath)) {
      const raw = fs.readFileSync(jsonPath, 'utf-8');
      const parsed = JSON.parse(raw);
      const list = Array.isArray(parsed?.models) ? parsed.models : Array.isArray(parsed) ? parsed : [];
      GOOD_MODELS_CACHE = list;
      return list;
    }
    const txtPath = path.resolve(process.cwd(), 'good_models.txt');
    if (fs.existsSync(txtPath)) {
      const raw = fs.readFileSync(txtPath, 'utf-8');
      const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#'));
      const list = lines.map((id) => ({ id }));
      GOOD_MODELS_CACHE = list;
      return list;
    }
  } catch (e) {
    console.warn('Failed to load good models list:', (e as Error).message);
  }
  GOOD_MODELS_CACHE = [];
  return [];
}

function lookupPricing(modelId: string): ModelPricing | undefined {
  const list = loadGoodModels();
  const needle = normalizeModelIdForLookup(modelId);
  const found = list.find(m => normalizeModelIdForLookup(m.id) === needle);
  if (!found) return undefined;
  return { inCostPerM: found.inCostPerM, outCostPerM: found.outCostPerM };
}

export class LLMService {
  private buildCleaningPrompt(
    text: string, 
    options: CleaningOptions,
    customInstructions?: string
  ): string {
    const tasks: string[] = [];

    if (options.replaceSmartQuotes) {
      tasks.push("- Replace all smart quotes (", ", ', ') with standard ASCII quotes (\", ')");
    }
    if (options.fixOcrErrors) {
      tasks.push("- Fix OCR errors: correct spacing issues and merged words (e.g., 'thebook' → 'the book')");
    }
    if (options.correctSpelling) {
      tasks.push("- Correct common spelling mistakes and typos");
    }
    if (options.removeUrls) {
      tasks.push("- Remove all URLs and web links");
    }
    if (options.removeFootnotes) {
      tasks.push("- Remove footnote markers (numbers, asterisks) and extraneous metadata");
    }
    if (options.addPunctuation) {
      tasks.push("- Add appropriate punctuation after headers and loose numbers for better TTS prosody");
    }

    let prompt = `You are a text cleaning assistant for TTS (text-to-speech) preprocessing. Your task is to clean and repair the following text.

Apply these transformations:
${tasks.join("\n")}

Important rules:
- Preserve the original meaning and content
- Only fix errors, don't rewrite or rephrase
- Maintain paragraph structure
- Return ONLY the cleaned text, no explanations`;

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    prompt += `\n\nText to clean:\n${text}\n\nCleaned text:`;

    return prompt;
  }

  private buildCleaningPromptConcise(
    text: string,
    options: CleaningOptions,
    customInstructions?: string
  ): string {
    const tasks: string[] = [];
    if (options.replaceSmartQuotes) tasks.push("- Replace smart quotes with ASCII");
    if (options.fixOcrErrors) tasks.push("- Fix OCR spacing and merged words");
    if ((options as any).fixHyphenation) tasks.push("- Fix hyphenation splits (merge words split by line breaks/hyphens)");
    if (options.correctSpelling) tasks.push("- Fix common spelling/typos");
    if (options.removeUrls) tasks.push("- Remove URLs");
    if (options.removeFootnotes) tasks.push("- Remove footnotes/metadata");
    if (options.addPunctuation) tasks.push("- Add punctuation after headers/loose numbers for TTS");

    let prompt = `You are a TTS text cleaner.\nApply only the selected transformations:\n${tasks.join("\n")}\nRules: preserve meaning, do not rewrite, keep paragraphs.\nReturn ONLY the cleaned text.`;

    if (customInstructions) {
      prompt += `\n\nCustom:\n${customInstructions}`;
    }

    prompt += `\n\nText:\n${text}\n\nCleaned:`;

    return prompt;
  }

  private buildSpeakerPrompt(
    text: string,
    config: SpeakerConfig,
    customInstructions?: string,
    extendedExamples?: boolean
  ): string {
    const labelExample =
      config.labelFormat === "speaker"
        ? "Speaker 1:, Speaker 2:, etc."
        : "[1]:, [2]:, [3]:, etc.";
    const narratorLabel = config.labelFormat === 'speaker' ? 'Speaker 1:' : '[1]:';

    let prompt = "";

    if (config.mode === "format") {
      prompt = `You are a dialogue formatting assistant. Convert the following text to a standardized multi-speaker format.

Requirements:
- Number of speakers: ${config.speakerCount}
- Label format: ${labelExample}
- Each speaker's line should start with their label
- Preserve all dialogue content exactly as written
- Only change the speaker label format`;
    } else {
      // Check if narrator is in the character mapping
      const hasNarratorFromMapping = config.characterMapping?.some(
        (char) => char.name.toLowerCase() === "narrator"
      );
      const hasNarrator = Boolean(config.includeNarrator || hasNarratorFromMapping);
      const narratorAttr = (config as any).narratorAttribution || 'remove';
      const narratorName = (config as any).narratorCharacterName && String((config as any).narratorCharacterName).trim() || undefined;
      // For prompts, reserve Speaker 1 for narration when present
      const narratorSpNum = narratorName && hasNarrator ? 1 : undefined;

      if (hasNarrator) {
        prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and structure it for multi-speaker TTS.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each speaking character's line as: ${labelExample} [dialogue text]
- ${narratorAttr === 'remove'
            ? 'Remove dialogue attribution tags (e.g., "he said", "she replied", "Bob asked"). Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these attribution phrases inline as part of the spoken line.'
            : narratorAttr === 'verbatim'
              ? 'Move dialogue attribution tags (e.g., "he said", "she replied", "Bob asked") into separate Narrator lines immediately following the spoken line (preserve punctuation). Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these phrases inline in the spoken line (do not move).'
              : 'Convert dialogue attribution and action context into a separate Narrator line immediately after the spoken line. Rewrite the tag and any action into a concise descriptive sentence without quotes (e.g., "John passed the book over to Tim."). Prefer explicit names over pronouns when ambiguous. Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these attribution phrases inline as part of the spoken line.'}
- Label narration (non-dialogue) using ${narratorLabel} (do NOT use the literal label "Narrator:")
 - Narrator speaking rule: When the narrator (Speaker 1) speaks, preserve first‑person speaking cues inline (e.g., "I said", "I asked", "I replied", "I whispered", "I shouted"). Do NOT remove, move, or convert these phrases; keep them attached to the dialogue so TTS switches to the speaking voice.
- Preserve full content and order from the input. Do not omit or reorder any parts.
- ${narratorName ? `Narrator identity: The narrator is the character "${narratorName}"${narratorSpNum ? ` (Speaker ${narratorSpNum})` : ''}. Keep narration as Narrator lines but write the narration content in first-person ("I") from that character's perspective. Do not use the narrator's name inside narration.` : 'If the text uses first-person narration ("I") that clearly refers to a speaking character, keep narration as Narrator lines and write from first-person ("I") rather than using the character\'s name. Use Narrator only for narration; use speaker labels for spoken dialogue.'}
- Assign consistent speaker numbers based on who speaks`;
      } else {
        prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and extract dialogue between ${config.speakerCount} speakers.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each line as: ${labelExample} [dialogue text]
- Assign consistent speaker numbers based on who speaks
- Extract only spoken dialogue; omit non-dialogue narration/descriptions.
- Remove dialogue attribution tags (e.g., "he said", "she replied")
- If a character name is detected, keep it consistent with one speaker number
- Preserve full content and order from the input. Do not omit or reorder any parts.
- ${narratorName ? `Narrator identity: The narrator is the character "${narratorName}"${narratorSpNum ? ` (Speaker ${narratorSpNum})` : ''}. Keep narration (non-dialogue) as Narrator lines and write them in first-person ("I"). Do not include the narrator's own name inside narration.` : 'If the story uses first-person narration ("I") clearly from a character\'s perspective, keep narration as Narrator lines and write from first-person ("I").'}
- If dialogue occurs from characters not in the known set, keep that dialogue unmodified and assign a new speaker number (do not label as narration).`;
      }
    }

    // Add character mapping if available (adjust so narration occupies Speaker 1)
    if (config.characterMapping && config.characterMapping.length > 0) {
      const narratorNameLower = ((config as any).narratorCharacterName || '').toString().trim().toLowerCase();
      const hasAnyNarrator = Boolean(config.includeNarrator || config.characterMapping.some(c => c.name.toLowerCase() === 'narrator'));
      const adjusted = (hasAnyNarrator ? config.characterMapping.map(c => ({
        name: c.name,
        speakerNumber: narratorNameLower && c.name.toLowerCase() === narratorNameLower ? 1 : (c.speakerNumber + 1),
      })) : config.characterMapping);
      const mappingList = adjusted
        .map((char) => `  - ${char.name} = Speaker ${char.speakerNumber}`)
        .join("\n");
      
      const hasNarratorInMapping = config.characterMapping.some((char) => char.name.toLowerCase() === "narrator");
      const narratorActive = Boolean(config.includeNarrator || hasNarratorInMapping);

      if (narratorActive) {
        const narratorAttr = (config as any).narratorAttribution || 'remove';
        prompt += `\n\nCharacter to Speaker Mapping (use these exact assignments):
${mappingList}

IMPORTANT: 
- Only extract dialogue from the speaking characters listed above
- Label narration (non-dialogue) with ${narratorLabel}
- ${narratorAttr === 'remove' ? 'Remove dialogue attribution tags like "he said", "she whispered", "Alice asked"' : narratorAttr === 'verbatim' ? 'Move dialogue attribution tags like "he said", "she whispered", "Alice asked" into a separate Narrator line placed immediately after the spoken line' : 'Transform dialogue attribution and actions into a separate Narrator line that summarizes the action (e.g., "John passed the book over to Tim.") placed immediately after the spoken line'}
- If a speaking character not listed in the mapping appears, keep their dialogue unmodified and attribute it to ${narratorLabel}`;
      } else {
        prompt += `\n\nCharacter to Speaker Mapping (use these exact assignments when possible):
${mappingList}

IMPORTANT:
- Prefer these assignments for known characters.
- If a speaking character not listed in the mapping appears, keep their dialogue unmodified and attribute it to ${narratorLabel}.`;
      }
    }

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    if (extendedExamples && config.mode === 'intelligent') {
      const exLabel = narratorLabel;
      prompt += `\n\nExamples (narrator speech handling):\n1) He was standing outside in the cold waiting for me. I said, "Let's go."\nExpected:\n${exLabel} I said, "Let's go."\n\n2) "Let's go," I said.\nExpected:\n${exLabel} "Let's go," I said.\n\n3) He was standing outside... I asked, "Are you ready?"\nExpected:\n${exLabel} I asked, "Are you ready?"\n\n4) Unknown speaker appears briefly in dialogue\nExpected: Keep the line, attribute to ${exLabel} without modification.`;
    }

    prompt += `\n\nText to format:\n${text}\n\nFormatted dialogue:`;

    return prompt;
  }

  private async runInference(prompt: string, options: ProcessChunkOptions): Promise<{ text: string; usage: InferenceUsage }> {
    const { modelSource = 'api', modelName, localModelName, ollamaModelName } = options;
    const resolvedModelName = normalizeHuggingFaceModelId(modelName);
    const pricing = modelSource === 'api' ? (lookupPricing(resolvedModelName) || undefined) : undefined;
    const inputTokens = estimateTokens(prompt);
    let output = '';

    if (modelSource === 'ollama') {
      const model = ollamaModelName || "llama3.1:8b";
      const url = `${OLLAMA_BASE_URL}/api/generate`;
      const isThinking = isThinkingOllamaModel(model);
      const options = buildOllamaOptions(
        {
          temperature: 0.3,
          num_predict: 2000,
          num_ctx: 8192,
        },
        modelSource,
        model
      );
      const init: RequestInit = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
          keep_alive: isThinking ? '15m' : '5m',
          options,
        }),
      };
      logRequest("Ollama.generate", url, init);
      const res = await fetch(url, init);
      // Try to capture a short response preview for debugging
      const clone = res.clone();
      try {
        const preview = await clone.text();
        logResponse("Ollama.generate", res, { bodyPreview: preview.length > 600 ? `${preview.slice(0, 600)}… [truncated ${preview.length - 600} chars]` : preview });
      } catch {}
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Ollama request failed (${res.status}): ${body}`);
      }
      const json: any = await res.json();
      output = json.response || json.final_response || json?.message?.content || '';
    } else {
      // Check if API token is available
      if (!apiToken) {
        throw new Error(
          'HuggingFace API token not found. Please set HUGGINGFACE_API_TOKEN in your environment (dotenv/.env), or switch to Ollama model source.'
        );
      }

      try {
        const isFireworks = HF_PROVIDER === 'fireworks-ai';
        if (isFireworks) {
          if (DEBUG_ENABLED) {
            console.debug('[LLM DEBUG] HF.chatCompletion (Fireworks) call', {
              provider: HF_PROVIDER,
              model: resolvedModelName,
              auth: apiToken?.startsWith('hf_') ? 'hf-token (router)' : 'none',
              promptPreview: prompt.length > 500 ? `${prompt.slice(0, 500)}… [truncated ${prompt.length - 500} chars]` : prompt,
              parameters: { max_tokens: 2000, temperature: 0.3 },
            });
          }
          const response = await hf.chatCompletion(
            {
              provider: 'fireworks-ai' as any,
              model: resolvedModelName,
              messages: [
                { role: 'user', content: prompt },
              ],
              temperature: 0.3,
              max_tokens: 2000,
            },
            {
              fetch: createLoggingFetch('HF.chatCompletion.Fireworks'),
            }
          );
          output = response.choices?.[0]?.message?.content || '';
        } else {
          // Default to text generation via selected provider (router). If provider doesn't support text-generation (e.g., fireworks-ai), fallback to chatCompletion.
          if (DEBUG_ENABLED) {
            console.debug('[LLM DEBUG] HF.textGeneration call', {
              provider: HF_PROVIDER,
              model: resolvedModelName,
              tokenPresent: Boolean(apiToken),
              promptPreview: prompt.length > 500 ? `${prompt.slice(0, 500)}… [truncated ${prompt.length - 500} chars]` : prompt,
              parameters: { max_new_tokens: 2000, temperature: 0.3, return_full_text: false },
            });
          }
          try {
            const response = await hf.textGeneration(
              {
                model: resolvedModelName,
                inputs: prompt,
                parameters: {
                  max_new_tokens: 2000,
                  temperature: 0.3,
                  return_full_text: false,
                },
                provider: HF_PROVIDER as any,
              },
              {
                // Inject logging fetch to capture final URL & headers
                fetch: createLoggingFetch('HF.textGeneration'),
              }
            );
            output = response.generated_text || '';
          } catch (e) {
            const err = e as Error;
            const prov = parseProviderFromTaskError(err.message);
            if (prov && supportsConversationalFromError(err.message)) {
              if (DEBUG_ENABLED) {
                console.debug('[LLM DEBUG] Falling back to HF.chatCompletion due to unsupported text-generation', { provider: prov });
              }
              const response = await hf.chatCompletion(
                {
                  provider: prov as any,
                  model: resolvedModelName,
                  messages: [ { role: 'user', content: prompt } ],
                  temperature: 0.3,
                  max_tokens: 2000,
                },
                { fetch: createLoggingFetch('HF.chatCompletion.fallback') }
              );
              output = response.choices?.[0]?.message?.content || '';
            } else {
              throw e;
            }
          }
        }
      } catch (error) {
        // Provide more helpful error messages
        if (error instanceof Error) {
          if (error.message.toLowerCase().includes('provider information') || error.message.toLowerCase().includes('inference provider')) {
            throw new Error(
              'Selected model is not available on the selected provider. Try a supported model like meta-llama/Llama-3.1-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.2, or set HF_PROVIDER=auto.'
            );
          }
          if (error.message.includes('401') || error.message.includes('unauthorized') || 
              error.message.includes('authentication') || error.message.includes('token')) {
            throw new Error(
              'HuggingFace API authentication failed. Your API token may be invalid or expired. Please check your HUGGINGFACE_API_TOKEN, or switch to Ollama.'
            );
          }
          if (DEBUG_ENABLED) {
            console.debug('[LLM DEBUG] HF.textGeneration error', {
              message: (error as Error).message,
              name: (error as Error).name,
            });
          }
        }
        throw error;
      }
    }
    const outputTokens = estimateTokens(output);
    const inCost = pricing?.inCostPerM ? (inputTokens / 1_000_000) * pricing.inCostPerM : 0;
    const outCost = pricing?.outCostPerM ? (outputTokens / 1_000_000) * pricing.outCostPerM : 0;
    return { text: output, usage: { inputTokens, outputTokens, inputCost: inCost, outputCost: outCost } };
  }

  private buildSinglePassPrompt(
    text: string,
    cleaning: CleaningOptions,
    config: SpeakerConfig,
    customInstructions?: string,
    extendedExamples?: boolean
  ): string {
    const labelExample =
      config.labelFormat === "speaker"
        ? "Speaker 1:, Speaker 2:, ..."
        : "[1]:, [2]:, [3]:, ...";
    const narratorLabel = config.labelFormat === 'speaker' ? 'Speaker 1:' : '[1]:';

    const parts: string[] = [];
    const cleanTasks: string[] = [];
    if (cleaning.replaceSmartQuotes) cleanTasks.push("replace smart quotes with ASCII");
    if (cleaning.fixOcrErrors) cleanTasks.push("fix OCR spacing/merged words");
    if ((cleaning as any).fixHyphenation) cleanTasks.push("fix hyphenation splits");
    if (cleaning.correctSpelling) cleanTasks.push("fix common spelling/typos");
    if (cleaning.removeUrls) cleanTasks.push("remove URLs");
    if (cleaning.removeFootnotes) cleanTasks.push("remove footnotes/metadata");
    if (cleaning.addPunctuation) cleanTasks.push("add punctuation after headers/loose numbers for TTS");
    parts.push(`CLEANING: ${cleanTasks.join("; ")}. Preserve meaning; no rewrites; keep paragraphs.`);

    const hasNarratorFromMapping = config.characterMapping?.some((c) => c.name.toLowerCase() === 'narrator');
    const hasNarrator = Boolean(config.includeNarrator || hasNarratorFromMapping);
    const narratorAttr = (config as any).narratorAttribution || 'remove';
    const narratorName = (config as any).narratorCharacterName && String((config as any).narratorCharacterName).trim() || undefined;
    const narratorSpNum = narratorName
      ? config.characterMapping?.find(c => c.name.toLowerCase() === narratorName.toLowerCase())?.speakerNumber
      : undefined;

    if (config.mode === 'format') {
      parts.push(`FORMAT: Convert to multi-speaker format. Labels: ${labelExample}. Only change labels; keep content.`);
    } else {
      if (hasNarrator) {
        const attrRule = narratorAttr === 'remove'
          ? 'Remove attribution tags (e.g., "he said"). Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these phrases inline in the spoken line.'
          : narratorAttr === 'verbatim'
            ? 'Move attribution tags into a Narrator line immediately after the spoken line. Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these phrases inline in the spoken line (do not move).'
            : 'Convert attribution/action into a concise Narrator line after the spoken line (e.g., "John passed the book to Tim."). Prefer explicit names when ambiguous. Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these phrases inline in the spoken line (do not convert).';
        parts.push(`INTELLIGENT DIALOGUE: Detect speakers; output ${labelExample} for spoken lines. ${attrRule} Label narration with ${narratorLabel} (do not use the literal label "Narrator:"). ${narratorName ? `Narrator identity: "${narratorName}"${narratorSpNum ? ` (Speaker ${narratorSpNum})` : ''}. Write narration in first-person ("I") from that character's perspective and do not use their name inside narration.` : 'If first-person narration ("I") clearly refers to a speaking character, label narration with ${narratorLabel} and write from first-person ("I").'} Preserve full content and order from the input; do not omit or reorder. Use consistent speaker numbers.`);
        parts.push(`NARRATOR SPEAKING RULE: When the narrator (Speaker 1) speaks, preserve first-person speaking cues inline (e.g., "I said", "I asked", "I replied", "I whispered", "I shouted"). Do NOT remove, move, or convert these phrases; keep them attached to the dialogue so TTS switches to the speaking voice.`);
      } else {
        parts.push(`INTELLIGENT DIALOGUE: Detect speakers; output ${labelExample} for spoken lines only. Remove attribution tags. If dialogue appears from an unknown/unmapped character, keep it unmodified and assign a new speaker number (do not label as narration). Preserve full content and order from the input; do not omit or reorder. Keep consistent speaker numbering.`);
      }
    }

    if (config.characterMapping && config.characterMapping.length > 0) {
      const narratorNameLower = ((config as any).narratorCharacterName || '').toString().trim().toLowerCase();
      const hasAnyNarrator = Boolean(config.includeNarrator || config.characterMapping.some(c => c.name.toLowerCase() === 'narrator'));
      const adjusted = (hasAnyNarrator ? config.characterMapping.map(c => ({
        name: c.name,
        speakerNumber: narratorNameLower && c.name.toLowerCase() === narratorNameLower ? 1 : (c.speakerNumber + 1),
      })) : config.characterMapping);
      const mappingList = adjusted.map((c) => `${c.name} = Speaker ${c.speakerNumber}`).join('; ');
      parts.push(`MAPPING: ${mappingList}. Use exactly these assignments. For any speaking character not listed, keep their dialogue unmodified and attribute it to ${narratorLabel}.`);
    }

    if (customInstructions) parts.push(`CUSTOM: ${customInstructions}`);
    if (extendedExamples && config.mode === 'intelligent') {
      const ex = [
        `EXAMPLES:`,
        `1) He was standing outside in the cold waiting for me. I said, "Let's go."`,
        `   Output: ${narratorLabel} I said, "Let's go."`,
        `2) "Let's go," I said.`,
        `   Output: ${narratorLabel} "Let's go," I said.`,
        `3) He was standing outside... I asked, "Are you ready?"`,
        `   Output: ${narratorLabel} I asked, "Are you ready?"`,
      ];
      parts.push(ex.join('\n'));
    }
    parts.push(`Return ONLY the cleaned, formatted output. No explanations.`);

    return `${parts.join('\n')}
\nTEXT:\n${text}\n\nOUTPUT:`;
  }

  async processChunk(options: ProcessChunkOptions): Promise<{ text: string; usage: InferenceUsage }> {
    const { text, cleaningOptions, speakerConfig, customInstructions } = options;
    const deterministic = applyDeterministicCleaning(text, cleaningOptions, "pre");
    const cleanedInput = deterministic.text;

    // Stage 1: Text cleaning
    let processedText = cleanedInput;
    let totalInTokens = 0, totalOutTokens = 0, totalInCost = 0, totalOutCost = 0;
    if (speakerConfig && speakerConfig.mode !== "none" && options.singlePass) {
      if (DEBUG_ENABLED) {
        console.debug('[LLM DEBUG] Single-pass processing enabled');
      }
      const singlePrompt = this.buildSinglePassPrompt(cleanedInput, cleaningOptions, speakerConfig, customInstructions, options.extendedExamples);
      const r = await this.runInference(singlePrompt, options);
      if (r.text) {
        processedText = r.text;
      }
      totalInTokens += r.usage.inputTokens; totalOutTokens += r.usage.outputTokens;
      totalInCost += r.usage.inputCost; totalOutCost += r.usage.outputCost;
    } else {
      const useConcise = options.concisePrompts !== false;
      const cleaningPrompt = useConcise
        ? this.buildCleaningPromptConcise(cleanedInput, cleaningOptions, customInstructions)
        : this.buildCleaningPrompt(cleanedInput, cleaningOptions, customInstructions);
      const r1 = await this.runInference(cleaningPrompt, options);
      if (r1.text) {
        processedText = r1.text;
      }
      totalInTokens += r1.usage.inputTokens; totalOutTokens += r1.usage.outputTokens;
      totalInCost += r1.usage.inputCost; totalOutCost += r1.usage.outputCost;
    }

    if (!processedText) {
      processedText = cleanedInput;
    }

    // Stage 2: Speaker formatting (if configured and mode is not "none")
    if (speakerConfig && speakerConfig.mode !== "none" && !options.singlePass) {
      const speakerPrompt = this.buildSpeakerPrompt(processedText, speakerConfig, customInstructions, options.extendedExamples);
      const r2 = await this.runInference(speakerPrompt, options);
      const stage2Text = r2.text;
      totalInTokens += r2.usage.inputTokens; totalOutTokens += r2.usage.outputTokens;
      totalInCost += r2.usage.inputCost; totalOutCost += r2.usage.outputCost;
      
      if (stage2Text) {
        processedText = stage2Text;
      }
    }

    const finalPass = applyDeterministicCleaning(processedText, cleaningOptions, "post");
    return {
      text: finalPass.text,
      usage: {
        inputTokens: totalInTokens,
        outputTokens: totalOutTokens,
        inputCost: totalInCost,
        outputCost: totalOutCost,
      }
    };
  }

  async validateOutput(
    originalText: string,
    processedText: string,
    modelSource?: ModelSource
  ): Promise<{ valid: boolean; issues?: string[] }> {
    // Simple validation: check if output is not empty and has reasonable length
    if (!processedText || processedText.length < originalText.length * 0.5) {
      return {
        valid: false,
        issues: ["Output too short or empty"],
      };
    }

    // Check for common error patterns
    const issues: string[] = [];

    if (processedText.includes("[ERROR]") || processedText.includes("I cannot")) {
      issues.push("Model returned error message");
    }

    if (processedText.split("\n").length < 2 && originalText.split("\n").length > 5) {
      issues.push("Lost paragraph structure");
    }

    // For local models, use lightweight validation only (no API calls)
    // For API models, we could add more sophisticated validation if needed
    
    return {
      valid: issues.length === 0,
      issues: issues.length > 0 ? issues : undefined,
    };
  }

  getPromptPreviews(
    sampleText: string,
    cleaningOptions: CleaningOptions,
    speakerConfig?: SpeakerConfig,
    customInstructions?: string,
    singlePass?: boolean,
    concisePrompts?: boolean,
    extendedExamples?: boolean
  ): { stage1: string; stage2?: string } {
    const stage1 = speakerConfig && speakerConfig.mode !== 'none' && singlePass
      ? this.buildSinglePassPrompt(sampleText, cleaningOptions, speakerConfig, customInstructions, extendedExamples)
      : (concisePrompts !== false
          ? this.buildCleaningPromptConcise(sampleText, cleaningOptions, customInstructions)
          : this.buildCleaningPrompt(sampleText, cleaningOptions, customInstructions));
    
    const result: { stage1: string; stage2?: string } = { stage1 };

    if (speakerConfig && speakerConfig.mode !== "none" && !singlePass) {
      result.stage2 = this.buildSpeakerPrompt(sampleText, speakerConfig, customInstructions, extendedExamples);
    }

    return result;
  }

  async extractCharacters(options: {
    text: string;
    includeNarrator: boolean;
    modelSource?: ModelSource;
    modelName: string;
    localModelName?: string;
    ollamaModelName?: string;
  }): Promise<{ characters: Array<{ name: string; speakerNumber: number }>; narratorCharacterName?: string }> {
    const { text, includeNarrator, modelSource = 'api', modelName, localModelName, ollamaModelName } = options;
    const resolvedModelName = normalizeHuggingFaceModelId(modelName);

    const prompt = `You are a character extraction assistant for multi-speaker TTS systems. Analyze the following text sample and extract character/speaker names and narrator identity.

Requirements:
- Extract only actual character names that speak in the text
- List each unique character only once
- Return names in order of first appearance
${includeNarrator ? "- If there is clear neutral narrative/descriptive text (third-person), you may include a Narrator identity. If the text uses first-person narration (\\\"I\\\") that clearly corresponds to a speaking character, do not add a separate 'Narrator' entry; instead set narratorIsCharacter to that character's name." : "- Do NOT include narrator/descriptive-only entries. If the text uses first-person narration (\\\"I\\\") that clearly corresponds to a speaking character, set narratorIsCharacter to that character's name (not 'Narrator')."}
- Return ONLY a JSON object with fields { characters: string[], narratorIsCharacter: string|null }, no explanations

Response JSON schema:
{ "characters": ["Character1", "Character2"], "narratorIsCharacter": "Character1" | null }

Text sample:
${text}

JSON only:`;

    let content: string;

    if (modelSource === 'ollama') {
      const model = ollamaModelName || "llama3.1:8b";
      const url = `${OLLAMA_BASE_URL}/api/generate`;
      const options = buildOllamaOptions(
        {
          temperature: 0.2,
          num_predict: 800,
          num_ctx: 6144,
        },
        modelSource,
        model
      );
      const isThinking = isThinkingOllamaModel(model);
      const init: RequestInit = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt, stream: false, keep_alive: isThinking ? '15m' : '5m', options }),
      };
      logRequest("Ollama.generate", url, init);
      const res = await fetch(url, init);
      const clone = res.clone();
      try {
        const preview = await clone.text();
        logResponse("Ollama.generate", res, { bodyPreview: preview.length > 600 ? `${preview.slice(0, 600)}… [truncated ${preview.length - 600} chars]` : preview });
      } catch {}
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Ollama request failed (${res.status}): ${body}`);
      }
      const json: any = await res.json();
      content = json.response || json.final_response || json?.message?.content || "[]";
    } else {
      // Check if API token is available
      if (!apiToken) {
        throw new Error(
          'HuggingFace API token not found. Please set HUGGINGFACE_API_TOKEN in your environment (dotenv/.env), or switch to Ollama model source.'
        );
      }

      try {
        const isFireworks = HF_PROVIDER === 'fireworks-ai';
        if (isFireworks) {
          if (DEBUG_ENABLED) {
            console.debug('[LLM DEBUG] HF.chatCompletion (extractCharacters, Fireworks) call', {
              provider: HF_PROVIDER,
              model: resolvedModelName,
              auth: apiToken?.startsWith('hf_') ? 'hf-token (router)' : 'none',
              promptPreview: prompt.length > 500 ? `${prompt.slice(0, 500)}… [truncated ${prompt.length - 500} chars]` : prompt,
              parameters: { max_tokens: 500, temperature: 0.2 },
            });
          }
          const response = await hf.chatCompletion(
            {
              provider: 'fireworks-ai' as any,
              model: resolvedModelName,
              messages: [
                { role: 'user', content: prompt },
              ],
              temperature: 0.2,
              max_tokens: 500,
            },
            { fetch: createLoggingFetch('HF.chatCompletion.extractCharacters.Fireworks') }
          );
          content = response.choices?.[0]?.message?.content || "[]";
        } else {
          // Use text generation for HF Inference; fallback to chatCompletion if provider doesn't support text-generation
          if (DEBUG_ENABLED) {
            console.debug('[LLM DEBUG] HF.textGeneration (extractCharacters) call', {
              provider: HF_PROVIDER,
              model: resolvedModelName,
              tokenPresent: Boolean(apiToken),
              promptPreview: prompt.length > 500 ? `${prompt.slice(0, 500)}… [truncated ${prompt.length - 500} chars]` : prompt,
              parameters: { max_new_tokens: 500, temperature: 0.2, return_full_text: false },
            });
          }
          try {
            const response = await hf.textGeneration(
              {
                model: resolvedModelName,
                inputs: prompt,
                parameters: {
                  max_new_tokens: 500,
                  temperature: 0.2,
                  return_full_text: false,
                },
                provider: HF_PROVIDER as any,
              },
              {
                fetch: createLoggingFetch('HF.textGeneration.extractCharacters'),
              }
            );
            content = response.generated_text || "[]";
          } catch (e) {
            const err = e as Error;
            const prov = parseProviderFromTaskError(err.message);
            if (prov && supportsConversationalFromError(err.message)) {
              if (DEBUG_ENABLED) {
                console.debug('[LLM DEBUG] Falling back to HF.chatCompletion (extractCharacters) due to unsupported text-generation', { provider: prov });
              }
              const response = await hf.chatCompletion(
                {
                  provider: prov as any,
                  model: resolvedModelName,
                  messages: [ { role: 'user', content: prompt } ],
                  temperature: 0.2,
                  max_tokens: 500,
                },
                { fetch: createLoggingFetch('HF.chatCompletion.extractCharacters.fallback') }
              );
              content = response.choices?.[0]?.message?.content || "[]";
            } else {
              throw e;
            }
          }
        }
      } catch (error) {
        // Provide more helpful error messages
        if (error instanceof Error) {
          if (error.message.toLowerCase().includes('provider information') || error.message.toLowerCase().includes('inference provider')) {
            throw new Error(
              'Selected model is not available on the selected provider. Try a supported model like meta-llama/Llama-3.1-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.2, or set HF_PROVIDER=auto.'
            );
          }
          if (error.message.includes('401') || error.message.includes('unauthorized') || 
              error.message.includes('authentication') || error.message.includes('token')) {
            throw new Error(
              'HuggingFace API authentication failed. Your API token may be invalid or expired. Please check your HUGGINGFACE_API_TOKEN, or switch to Ollama.'
            );
          }
          if (DEBUG_ENABLED) {
            console.debug('[LLM DEBUG] HF.textGeneration (extractCharacters) error', {
              message: (error as Error).message,
              name: (error as Error).name,
            });
          }
        }
        throw error;
      }
    }
    
    // Parse JSON. Support either the new object format or legacy array format.
    let parsed: any;
    let characterNames: string[] = [];
    let narratorCharacterName: string | undefined = undefined;
    try {
      parsed = JSON.parse(content.trim());
    } catch {
      // Try to extract JSON code block/object first
      const objMatch = content.match(/\{[\s\S]*\}/);
      if (objMatch) {
        try { parsed = JSON.parse(objMatch[0]); } catch {}
      }
      if (!parsed) {
        const arrMatch = content.match(/\[[\s\S]*?\]/);
        if (arrMatch) {
          try { parsed = JSON.parse(arrMatch[0]); } catch {}
        }
      }
    }

    if (Array.isArray(parsed)) {
      characterNames = parsed;
    } else if (parsed && typeof parsed === 'object') {
      if (Array.isArray(parsed.characters)) characterNames = parsed.characters;
      const narr = parsed.narratorIsCharacter;
      if (typeof narr === 'string' && narr.trim()) narratorCharacterName = narr.trim();
    }
    if (!Array.isArray(characterNames)) characterNames = [];

    const characters = characterNames.map((name, index) => ({
      name: String(name).trim(),
      speakerNumber: index + 1,
    }));
    return { characters, narratorCharacterName };
  }
}

export const llmService = new LLMService();
