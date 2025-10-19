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
const LLM_DEBUG_FULL = String(process.env.LLM_DEBUG_FULL || "").toLowerCase();
const DEBUG_FULL_BODY = LLM_DEBUG_FULL === "1" || LLM_DEBUG_FULL === "true" || LLM_DEBUG_FULL === "yes";
const LOG_FILE_ENV = (process.env.LLM_DEBUG_FILE || "").trim();
const DEFAULT_LOG_FILE = path.resolve(process.cwd(), "logs", "llm-debug.txt");
const FALLBACK_LOG_FILE = path.resolve(process.cwd(), "temp_llm.txt");

function ensureLogPath(): string | null {
  if (!DEBUG_ENABLED) return null;
  try {
    const chosen = LOG_FILE_ENV || DEFAULT_LOG_FILE;
    const dir = path.dirname(chosen);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    return chosen;
  } catch {
    try { return FALLBACK_LOG_FILE; } catch { return null; }
  }
}

const DEBUG_LOG_PATH = ensureLogPath();

function writeDebugLine(label: string, payload?: unknown) {
  if (!DEBUG_ENABLED || !DEBUG_LOG_PATH) return;
  try {
    const ts = new Date().toISOString();
    const line = payload === undefined
      ? `[${ts}] ${label}\n`
      : `[${ts}] ${label} ${JSON.stringify(payload)}\n`;
    fs.appendFileSync(DEBUG_LOG_PATH, line, { encoding: "utf-8" });
  } catch {}
}

function writeDebugBlock(label: string, text: string) {
  if (!DEBUG_ENABLED || !DEBUG_LOG_PATH) return;
  try {
    const ts = new Date().toISOString();
    const header = `\n[${ts}] ${label} BEGIN\n`;
    const footer = `\n[${ts}] ${label} END\n`;
    fs.appendFileSync(DEBUG_LOG_PATH, header + text + footer, { encoding: "utf-8" });
  } catch {}
}

// Template utilities
function readTemplateFile(name: string): string | null {
  try {
    const p = path.resolve(process.cwd(), name);
    if (fs.existsSync(p)) {
      return fs.readFileSync(p, 'utf-8');
    }
  } catch {}
  return null;
}

function renderTemplate(tpl: string, vars: Record<string, string | undefined>): string {
  let out = tpl;
  for (const [key, val] of Object.entries(vars)) {
    const safe = (val ?? '').toString();
    out = out.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), safe);
  }
  return out;
}

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
  writeDebugLine(`${label} request`, { method, url: String(url), headers, body });
  try {
    if (DEBUG_FULL_BODY && typeof (init?.body) === 'string') {
      writeDebugBlock(`[HTTP] ${label} request body`, String(init?.body));
    }
  } catch {}
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
  writeDebugLine(`${label} response`, {
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
        if (DEBUG_FULL_BODY) {
          try { writeDebugBlock(`[HTTP] ${label} response body`, text); } catch {}
        }
      } else {
        logResponse(label, res);
      }
    } catch {
      logResponse(label, res);
    }
    return res;
  };
}

// Sanitize LLM model outputs (remove fences, thinking tags, echoed labels)
function stripCodeFences(s: string): string {
  let out = s.trim();
  out = out.replace(/^\s*```[a-zA-Z0-9_+\-]*\s*/i, "");
  out = out.replace(/\s*```\s*$/i, "");
  out = out.replace(/^\s*"""\s*/i, "");
  out = out.replace(/\s*"""\s*$/i, "");
  return out.trim();
}

function stripThinkingTags(s: string): string {
  let out = s;
  const tags = ["think", "thinking", "reflection", "reasoning", "scratchpad", "chain-of-thought", "cot", "c-o-t"];
  for (const t of tags) {
    const re = new RegExp(`<${t}>[\\s\\S]*?<\\/${t}>`, "gi");
    out = out.replace(re, "");
  }
  out = out.replace(/^(?:\s*(Thought|Reasoning|Chain of Thought)\s*:\s.*(?:\r?\n|$))+?/gmi, "");
  return out;
}

function stripLeadingLabels(s: string): string {
  return s.replace(/^(Cleaned\s*text|Formatted\s*dialogue|Formatted\s*dialog|Output|OUTPUT|Result|Cleaned|JSON\s*only)\s*:\s*/i, "").trim();
}

function extractAfterLastOutputCue(s: string): string {
  const cues = ["Cleaned text:", "Formatted dialogue:", "Formatted dialog:", "OUTPUT:", "Output:", "Cleaned:", "Result:"];
  let idx = -1;
  for (const cue of cues) {
    const i = s.lastIndexOf(cue);
    if (i > idx) idx = i + cue.length;
  }
  if (idx > -1 && idx < s.length) {
    return s.slice(idx).trim();
  }
  return s.trim();
}

function sanitizeModelOutput(raw: string, _promptHint?: string): string {
  if (!raw) return "";
  let out = raw.replace(/^\uFEFF/, "");
  out = stripCodeFences(out);
  out = stripThinkingTags(out);
  out = stripLeadingLabels(out);
  out = extractAfterLastOutputCue(out);
  return out.trim();
}

function normalizeHuggingFaceModelId(model: string): string {
  // Fix common Llama 3.1 naming: meta-llama/Meta-Llama-3.1-* -> meta-llama/Llama-3.1-*
  if (/^meta-llama\/Meta-Llama-3\.1-/i.test(model)) {
    return model.replace(/meta-llama\/Meta-Llama-3\.1-/i, 'meta-llama/Llama-3.1-');
  }
  return model;
}

function parseProviderFromTaskError(message: string): string | undefined {
  const patterns = [
    /Task ['"]?text[-_ ]generation['"]? not supported for provider ['"]?([^'"\s]+)['"]?/i,
    /provider ['"]?([^'"\s]+)['"]? does not support task ['"]?text[-_ ]generation['"]?/i,
    /text[-_ ]generation (?:isn't|is not|not) (?:supported|available) for provider ['"]?([^'"\s]+)['"]?/i,
  ];
  for (const re of patterns) {
    const m = message.match(re);
    if (m && m[1]) return m[1];
  }
  return undefined;
}

function supportsConversationalFromError(message: string): boolean {
  return (
    /Available tasks:\s*(?:.*\bconversational\b|.*\bchat[-_ ]?completion\b)/i.test(message) ||
    /Supported tasks:\s*(?:.*\bconversational\b|.*\bchat[-_ ]?completion\b)/i.test(message) ||
    /Try ['"]?(?:conversational|chat[-_ ]?completion)['"]?/i.test(message)
  );
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
  temperature?: number;
  customInstructions?: string;
  singlePass?: boolean;
  // When true, skip LLM-based cleaning and rely on local deterministic cleaning only
  llmCleaningDisabled?: boolean;
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
    if (options.replaceSmartQuotes) tasks.push(`* Replace all smart quotes (“ ” ‘ ’) with standard ASCII quotes (" and ').`);
    if (options.fixOcrErrors) tasks.push(`* Fix common OCR errors, such as spacing issues and merged words (e.g., 'thebook' -> 'the book').`);
    if ((options as any).fixHyphenation) tasks.push(`* Fix hyphenation splits (merge words split by line breaks/hyphens).`);
    if (options.correctSpelling) tasks.push(`* Correct common spelling mistakes and typos.`);
    if (options.removeUrls) tasks.push(`* Remove all URLs, web links, and email addresses.`);
    if (options.removeFootnotes) tasks.push(`* Remove footnote markers (e.g., numbers, asterisks) and any other extraneous metadata.`);
    if (options.addPunctuation) tasks.push(`* Ensure appropriate punctuation (like a period) follows any headers or loose numbers for better TTS prosody.`);

    let prompt = `You are a TTS preprocessing assistant. Clean and repair the text using ONLY the listed transformations.

Preprocessing Steps:
${tasks.join('\n')}

Rules:
- Preserve the original meaning and content.
- Only fix errors; do not rewrite or rephrase.
- Maintain paragraph structure.
- Return ONLY the cleaned text, no explanations.`;

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    prompt += `\n\nText:\n${text}\n\nCleaned:`;
    return prompt;
  }

  // Note: concise prompt variant removed in favor of stronger, explicit instructions.

  private buildSpeakerPrompt(
    text: string,
    config: SpeakerConfig,
    customInstructions?: string,
    extendedExamples?: boolean
  ): string {
    // Prefer sample_prompt.md template when available
    const tpl = readTemplateFile('sample_prompt.md');
    const mapping = (config.characterMapping && config.characterMapping.length > 0)
      ? config.characterMapping.map((c) => `${c.name} = Speaker ${c.speakerNumber}`).join('; ')
      : 'none';
    // Do not include any preprocessing/transformation instructions in Stage 2
    const preprocessing = '';
    const includeNarrator = Boolean(config.includeNarrator);
    const narratorName = (config as any).narratorCharacterName && String((config as any).narratorCharacterName).trim() || undefined;
    const labelFormat = (config.labelFormat || 'speaker');
    const speakerLabelExample = labelFormat === 'bracket' ? '[1]:' : 'Speaker 1:';
    const speakerLabelInstructions = labelFormat === 'bracket'
      ? 'Identify all unique speaking characters. Assign them labels dynamically using bracket format: the first character to speak becomes [1]:, the second becomes [2]:, and so on.'
      : 'Identify all unique speaking characters. Assign them labels dynamically: the first character to speak becomes Speaker 1:, the second becomes Speaker 2:, and so on.';
    const narratorRule = includeNarrator
      ? 'All non‑quoted narrative, descriptive, or action text MUST be labeled using the Narrator: tag — never as any Speaker X:. Do not attribute narration content to speakers.'
      : 'Omit non‑quoted narrative, descriptive, or action text. Output only spoken dialogue labeled with the chosen speaker format.';
    const narratorIdentity = narratorName
      ? `Narrator Identity: The narrator is "${narratorName}". For narration, write from first‑person ("I") from that character’s perspective. Do not include the narrator’s name inside narration.`
      : '';
    const narratorAttr = (config as any).narratorAttribution || 'contextual';
    let attributionRule: string;
    if (config.mode === 'format') {
      attributionRule = 'Do not transform attribution tags. Preserve the original text and punctuation. Only add speaker labels and remove surrounding quotation marks for dialogue.';
    } else {
      attributionRule = narratorAttr === 'remove'
        ? 'Remove simple attribution tags (e.g., he said, Alice asked) entirely. The Speaker label makes them redundant.'
        : narratorAttr === 'verbatim'
          ? 'Move attribution tags (e.g., he said, Alice asked) into a separate Narrator: line immediately after the spoken line (preserve punctuation).'
          : 'Transform attribution and action: if attribution is paired with action, convert it into a concise Narrator: line (omit the attribution verb); if attribution is simple and provides no new action, omit it. Preserve the first‑person narrator exception.';
    }

    // Examples: prefer actual mapped names to guide the model
    let examples = '';
    if (extendedExamples && config.mode === 'intelligent') {
      const names = (config.characterMapping || []).map(c => c.name).filter(Boolean);
      const n1 = names[0] || 'Alice';
      const n2 = names[1] || 'Bob';
      const n3 = names[2] || 'Charlie';
      const exSpeaker1 = speakerLabelExample;
      const exSpeaker2 = labelFormat === 'bracket' ? '[2]:' : 'Speaker 2:';
      const ex: string[] = [];
      ex.push('Examples:');
      ex.push(`Input: "Are you coming to the party?" ${n1} asked.`);
      ex.push(`Output: ${exSpeaker1} Are you coming to the party?`);
      ex.push(`Input: "It's a beautiful day," ${n2} said, looking up at the sky.`);
      ex.push(`Output: ${exSpeaker2} It's a beautiful day.`);
      ex.push('Narrator: ' + `${n2} looked up at the sky.`);
      if (includeNarrator) {
        ex.push(`Input: I nodded in agreement.`);
        ex.push(`Output: Narrator: I nodded in agreement.`);
      }
      examples = ex.join('\n');
    }

    if (tpl) {
      return renderTemplate(tpl, {
        mapping,
        preprocessing,
        text,
        speaker_label_example: speakerLabelExample,
        speaker_label_instructions: speakerLabelInstructions,
        narrator_rule: narratorRule,
        narrator_identity: narratorIdentity,
        attribution_rule: attributionRule,
        examples,
      });
    }

    // Fallback inline prompt if template missing
    const parts: string[] = [];
    parts.push('You are a dialogue structuring assistant for multi-speaker TTS. Format into strict Speaker/Narrator lines.');
    parts.push(speakerLabelInstructions);
    parts.push(narratorRule);
    parts.push('Remove quotation marks from dialogue.');
    parts.push('Merge dialogue from the same speaker when separated only by an attribution.');
    parts.push(attributionRule);
    if (narratorIdentity) parts.push(narratorIdentity);
    parts.push(`\nText:\n${text}\n\nFormatted:`);
    return parts.join('\n');
  }

  private async runInference(prompt: string, options: ProcessChunkOptions): Promise<{ text: string; usage: InferenceUsage }> {
    const { modelSource = 'api', modelName, localModelName, ollamaModelName, temperature } = options;
    const resolvedModelName = normalizeHuggingFaceModelId(modelName);
    const pricing = modelSource === 'api' ? (lookupPricing(resolvedModelName) || undefined) : undefined;
    const inputTokens = estimateTokens(prompt);
    let output = '';

    if (DEBUG_ENABLED) {
      writeDebugBlock('INFERENCE PROMPT', prompt);
      writeDebugLine('INFERENCE META', { modelSource, modelName, localModelName, ollamaModelName, inputTokens });
    }

    const clampTemp = (t: number | undefined, def: number) => {
      const v = typeof t === 'number' && Number.isFinite(t) ? t : def;
      return Math.max(0, Math.min(2, v));
    };
    const tempGen = clampTemp(temperature as any, 0.3);

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
        model,
        // User-set temperature should override thinking tuning
        (typeof temperature === 'number' ? { temperature: tempGen } : undefined)
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
      if (!output || !String(output).trim()) {
        // Some chat-centric models behave better with /api/chat; retry once
        try {
          writeDebugLine('OLLAMA GENERATE EMPTY; RETRY CHAT', { model });
          const chatUrl = `${OLLAMA_BASE_URL}/api/chat`;
          const chatInit: RequestInit = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model,
              messages: [ { role: 'user', content: prompt } ],
              stream: false,
              keep_alive: isThinking ? '15m' : '5m',
              options,
            }),
          };
          logRequest("Ollama.chat", chatUrl, chatInit);
          const chatRes = await fetch(chatUrl, chatInit);
          const chatClone = chatRes.clone();
          try {
            const preview = await chatClone.text();
            logResponse("Ollama.chat", chatRes, { bodyPreview: preview.length > 600 ? `${preview.slice(0, 600)}… [truncated ${preview.length - 600} chars]` : preview });
            if (DEBUG_FULL_BODY) { try { writeDebugBlock('[HTTP] Ollama.chat response body', preview); } catch {} }
          } catch {}
          if (chatRes.ok) {
            const chatJson: any = await chatRes.json();
            output = chatJson?.message?.content || '';
          }
        } catch {}
      }
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
              temperature: tempGen,
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
              parameters: { max_new_tokens: 2000, temperature: tempGen, return_full_text: false },
            });
          }
          try {
            const response = await hf.textGeneration(
              {
                model: resolvedModelName,
                inputs: prompt,
                parameters: {
                  max_new_tokens: 2000,
                  temperature: tempGen,
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
            const prov = parseProviderFromTaskError(err.message) || HF_PROVIDER || 'auto';
            if (DEBUG_ENABLED) {
              console.debug('[LLM DEBUG] Falling back to HF.chatCompletion after textGeneration error', {
                provider: prov,
                message: err.message,
              });
            }
            const response = await hf.chatCompletion(
              {
                provider: prov as any,
                model: resolvedModelName,
                messages: [ { role: 'user', content: prompt } ],
                temperature: tempGen,
                max_tokens: 2000,
              },
              { fetch: createLoggingFetch('HF.chatCompletion.fallback') }
            );
            output = response.choices?.[0]?.message?.content || '';
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
    const cleaned = sanitizeModelOutput(output, prompt);
    if (DEBUG_ENABLED) {
      writeDebugBlock('INFERENCE RAW OUTPUT', output || '[empty]');
      writeDebugBlock('INFERENCE SANITIZED OUTPUT', cleaned || '[empty]');
    }
    const outputTokens = estimateTokens(cleaned);
    const inCost = pricing?.inCostPerM ? (inputTokens / 1_000_000) * pricing.inCostPerM : 0;
    const outCost = pricing?.outCostPerM ? (outputTokens / 1_000_000) * pricing.outCostPerM : 0;
    return { text: cleaned, usage: { inputTokens, outputTokens, inputCost: inCost, outputCost: outCost } };
  }

  private buildSinglePassPrompt(
    text: string,
    cleaning: CleaningOptions,
    config: SpeakerConfig,
    customInstructions?: string,
    extendedExamples?: boolean,
    llmCleaningDisabled?: boolean
  ): string {
    const tpl = readTemplateFile('sample_prompt.md');
    const mapping = (config.characterMapping && config.characterMapping.length > 0)
      ? config.characterMapping.map((c) => `${c.name} = Speaker ${c.speakerNumber}`).join('; ')
      : 'none';
    let preprocessing = '';
    if (llmCleaningDisabled) {
      preprocessing = 'Preprocessing Steps: (skip — already applied locally)';
    } else {
      const parts: string[] = [];
      if (cleaning.replaceSmartQuotes) parts.push(`* Replace smart quotes with ASCII (" and ')`);
      if (cleaning.fixOcrErrors) parts.push(`* Fix OCR errors: spacing issues and merged words.`);
      if ((cleaning as any).fixHyphenation) parts.push(`* Fix hyphenation splits.`);
      if (cleaning.correctSpelling) parts.push(`* Correct common spelling mistakes and typos.`);
      if (cleaning.removeUrls) parts.push(`* Remove URLs, links, and email addresses.`);
      if (cleaning.removeFootnotes) parts.push(`* Remove footnote markers and extraneous metadata.`);
      if (cleaning.addPunctuation) parts.push(`* Ensure punctuation after headers/loose numbers for TTS.`);
      preprocessing = ['Preprocessing Steps:', ...parts].join('\n');
    }

    const includeNarrator = Boolean(config.includeNarrator);
    const narratorName = (config as any).narratorCharacterName && String((config as any).narratorCharacterName).trim() || undefined;
    const labelFormat = (config.labelFormat || 'speaker');
    const speakerLabelExample = labelFormat === 'bracket' ? '[1]:' : 'Speaker 1:';
    const speakerLabelInstructions = labelFormat === 'bracket'
      ? 'Identify all unique speaking characters. Assign them labels dynamically using bracket format: the first character to speak becomes [1]:, the second becomes [2]:, and so on.'
      : 'Identify all unique speaking characters. Assign them labels dynamically: the first character to speak becomes Speaker 1:, the second becomes Speaker 2:, and so on.';
    const narratorRule = includeNarrator
      ? 'All non‑quoted narrative, descriptive, or action text MUST be labeled using the Narrator: tag — never as any Speaker X:. Do not attribute narration content to speakers.'
      : 'Omit non‑quoted narrative, descriptive, or action text. Output only spoken dialogue labeled with the chosen speaker format.';
    const narratorIdentity = narratorName
      ? `Narrator Identity: The narrator is "${narratorName}". For narration, write from first‑person ("I") from that character’s perspective. Do not include the narrator’s name inside narration.`
      : '';
    const narratorAttr = (config as any).narratorAttribution || 'contextual';
    let attributionRule: string;
    if (config.mode === 'format') {
      attributionRule = 'Do not transform attribution tags. Preserve the original text and punctuation. Only add speaker labels and remove surrounding quotation marks for dialogue.';
    } else {
      attributionRule = narratorAttr === 'remove'
        ? 'Remove simple attribution tags (e.g., he said, Alice asked) entirely. The Speaker label makes them redundant.'
        : narratorAttr === 'verbatim'
          ? 'Move attribution tags (e.g., he said, Alice asked) into a separate Narrator: line immediately after the spoken line (preserve punctuation).'
          : 'Transform attribution and action: if attribution is paired with action, convert it into a concise Narrator: line (omit the attribution verb); if attribution is simple and provides no new action, omit it. Preserve the first‑person narrator exception.';
    }

    let examples = '';
    if (extendedExamples && config.mode === 'intelligent') {
      examples = [
        'Examples:',
        `Input: "Where are we going?" she whispered as she picked up a map.`,
        `Output: ${speakerLabelExample} Where are we going?`,
        'Narrator: She whispered as she picked up a map.',
        `Input: "To the park," I replied.`,
        `Output: ${speakerLabelExample.replace('1', '2')} To the park, I replied.`
      ].join('\n');
    }

    if (tpl) {
      return renderTemplate(tpl, {
        mapping,
        preprocessing,
        text,
        speaker_label_example: speakerLabelExample,
        speaker_label_instructions: speakerLabelInstructions,
        narrator_rule: narratorRule,
        narrator_identity: narratorIdentity,
        attribution_rule: attributionRule,
        examples,
      });
    }
    return this.buildSpeakerPrompt(text, config, customInstructions, extendedExamples);
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
        writeDebugLine('PROCESS SINGLE-PASS', { enabled: true });
      }
      const singlePrompt = this.buildSinglePassPrompt(
        cleanedInput,
        cleaningOptions,
        speakerConfig,
        customInstructions,
        options.extendedExamples,
        options.llmCleaningDisabled === true
      );
      if (DEBUG_ENABLED) writeDebugBlock('SINGLE-PASS PROMPT', singlePrompt);
      const r = await this.runInference(singlePrompt, options);
      if (r.text) {
        processedText = r.text;
      }
      totalInTokens += r.usage.inputTokens; totalOutTokens += r.usage.outputTokens;
      totalInCost += r.usage.inputCost; totalOutCost += r.usage.outputCost;
    } else {
      if (options.llmCleaningDisabled === true) {
        // Skip LLM cleaning; rely on local deterministic cleaning
        if (DEBUG_ENABLED) writeDebugLine('CLEANING', { mode: 'local-only' });
      } else {
        const cleaningPrompt = this.buildCleaningPrompt(cleanedInput, cleaningOptions, customInstructions);
        if (DEBUG_ENABLED) writeDebugBlock('CLEANING PROMPT', cleaningPrompt);
        const r1 = await this.runInference(cleaningPrompt, options);
        if (r1.text) {
          processedText = r1.text;
        }
        totalInTokens += r1.usage.inputTokens; totalOutTokens += r1.usage.outputTokens;
        totalInCost += r1.usage.inputCost; totalOutCost += r1.usage.outputCost;
      }
    }

    if (!processedText) {
      processedText = cleanedInput;
    }

    // Stage 2: Speaker formatting (if configured and mode is not "none")
    if (speakerConfig && speakerConfig.mode !== "none" && !options.singlePass) {
      const speakerPrompt = this.buildSpeakerPrompt(processedText, speakerConfig, customInstructions, options.extendedExamples);
      if (DEBUG_ENABLED) writeDebugBlock('SPEAKER PROMPT', speakerPrompt);
      const r2 = await this.runInference(speakerPrompt, options);
      const stage2Text = r2.text;
      totalInTokens += r2.usage.inputTokens; totalOutTokens += r2.usage.outputTokens;
      totalInCost += r2.usage.inputCost; totalOutCost += r2.usage.outputCost;
      
      if (stage2Text) {
        processedText = stage2Text;
      }
    }

    const finalPass = applyDeterministicCleaning(processedText, cleaningOptions, "post");
    if (DEBUG_ENABLED) {
      writeDebugLine('PROCESS USAGE', { inputTokens: totalInTokens, outputTokens: totalOutTokens, inputCost: totalInCost, outputCost: totalOutCost });
      writeDebugBlock('PROCESS OUTPUT', finalPass.text);
    }
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
    extendedExamples?: boolean,
    llmCleaningDisabled?: boolean
  ): { stage1: string; stage2?: string } {
    const stage1 = speakerConfig && speakerConfig.mode !== 'none' && singlePass
      ? this.buildSinglePassPrompt(sampleText, cleaningOptions, speakerConfig, customInstructions, extendedExamples, llmCleaningDisabled === true)
      : this.buildCleaningPrompt(sampleText, cleaningOptions, customInstructions);
    
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
    temperature?: number;
  }): Promise<{ characters: Array<{ name: string; speakerNumber: number }>; narratorCharacterName?: string }> {
    const { text, includeNarrator, modelSource = 'api', modelName, localModelName, ollamaModelName, temperature } = options;
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

    if (DEBUG_ENABLED) {
      writeDebugBlock('CHARACTER EXTRACTION PROMPT', prompt);
    }

    const clampTemp = (t: number | undefined, def: number) => {
      const v = typeof t === 'number' && Number.isFinite(t) ? t : def;
      return Math.max(0, Math.min(2, v));
    };
    const tempExtract = clampTemp(temperature as any, 0.2);

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
        model,
        (typeof temperature === 'number' ? { temperature: tempExtract } : undefined)
      );
      const isThinking = isThinkingOllamaModel(model);
      const init: RequestInit = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt, stream: false, keep_alive: isThinking ? '15m' : '5m', options, format: 'json' }),
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
      content = json.response || json.final_response || json?.message?.content || "";
      if (!content || !String(content).trim()) {
        const think = typeof json.thinking === 'string' ? json.thinking : undefined;
        if (think && think.trim()) {
          writeDebugLine('OLLAMA EMPTY RESPONSE; USING THINKING FALLBACK', { bytes: think.length });
          content = sanitizeModelOutput(think);
        }
      }
      if (!content || !String(content).trim()) {
        content = "[]";
      }
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
              temperature: tempExtract,
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
              parameters: { max_new_tokens: 500, temperature: tempExtract, return_full_text: false },
            });
          }
          try {
            const response = await hf.textGeneration(
              {
                model: resolvedModelName,
                inputs: prompt,
                parameters: {
                  max_new_tokens: 500,
                  temperature: tempExtract,
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
            const prov = parseProviderFromTaskError(err.message) || HF_PROVIDER || 'auto';
            if (DEBUG_ENABLED) {
              console.debug('[LLM DEBUG] Falling back to HF.chatCompletion (extractCharacters) after textGeneration error', { provider: prov, message: err.message });
            }
            const response = await hf.chatCompletion(
              {
                provider: prov as any,
                model: resolvedModelName,
                messages: [ { role: 'user', content: prompt } ],
                temperature: tempExtract,
                max_tokens: 500,
              },
              { fetch: createLoggingFetch('HF.chatCompletion.extractCharacters.fallback') }
            );
            content = response.choices?.[0]?.message?.content || "[]";
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
    
    // Sanitize and log raw content from extraction
    const rawContent = content || "";
    const sanitizedContent = sanitizeModelOutput(rawContent);
    if (DEBUG_ENABLED) {
      writeDebugBlock('CHARACTER EXTRACTION RAW', rawContent || '[empty]');
      writeDebugBlock('CHARACTER EXTRACTION SANITIZED', sanitizedContent || '[empty]');
    }

    // Parse JSON. Support either the new object format or legacy array format.
    let parsed: any;
    let characterNames: string[] = [];
    let narratorCharacterName: string | undefined = undefined;
    try {
      parsed = JSON.parse(sanitizedContent.trim());
    } catch {
      // Try to extract JSON code block/object first
      const objMatch = sanitizedContent.match(/\{[\s\S]*\}/);
      if (objMatch) {
        try { parsed = JSON.parse(objMatch[0]); } catch {}
      }
      if (!parsed) {
        const arrMatch = sanitizedContent.match(/\[[\s\S]*?\]/);
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

    // Deduplicate characters while preserving order; exclude literal 'Narrator'
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const n of characterNames) {
      const nm = String(n).trim();
      if (!nm) continue;
      if (/^narrator$/i.test(nm)) continue;
      const key = nm.toLowerCase();
      if (!seen.has(key)) { seen.add(key); ordered.push(nm); }
    }
    const characters = ordered.map((name, index) => ({ name, speakerNumber: index + 1 }));
    if (DEBUG_ENABLED) {
      writeDebugLine('CHARACTER EXTRACTION RESULT', { characters, narratorCharacterName });
    }
    return { characters, narratorCharacterName };
  }
}

export const llmService = new LLMService();
