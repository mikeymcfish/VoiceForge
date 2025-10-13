import { HfInference } from "@huggingface/inference";
import type { CleaningOptions, SpeakerConfig, ModelSource } from "@shared/schema";

// Check if HuggingFace API token is available
const apiToken = process.env.HUGGINGFACE_API_TOKEN;
if (!apiToken) {
  console.warn("⚠️  HUGGINGFACE_API_TOKEN not found in environment variables!");
  console.warn("   API mode will not work. Please either:");
  console.warn("   1. Add HUGGINGFACE_API_TOKEN to your environment (dotenv/.env)");
  console.warn("   2. Use Ollama as the model source");
}

const hf = new HfInference(apiToken);
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
    customInstructions?: string
  ): string {
    const labelExample =
      config.labelFormat === "speaker"
        ? "Speaker 1:, Speaker 2:, etc."
        : "[1]:, [2]:, [3]:, etc.";

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

      if (hasNarrator) {
        prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and structure it for multi-speaker TTS.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each speaking character's line as: ${labelExample} [dialogue text]
- ${narratorAttr === 'remove' ? 'Remove dialogue attribution tags (e.g., "he said", "she replied", "Bob asked")' : narratorAttr === 'verbatim' ? 'Move dialogue attribution tags (e.g., "he said", "she replied", "Bob asked") into separate Narrator lines immediately following the spoken line (preserve punctuation).' : 'Convert dialogue attribution and action context into a separate Narrator line immediately after the spoken line. Rewrite the tag and any action into a concise descriptive sentence without quotes (e.g., "John passed the book over to Tim."). Prefer explicit names over pronouns when ambiguous.'}
- Preserve narrative descriptions and non-dialogue text as Narrator lines
- Assign consistent speaker numbers based on who speaks`;
      } else {
        prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and extract dialogue between ${config.speakerCount} speakers.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each line as: ${labelExample} [dialogue text]
- Assign consistent speaker numbers based on who speaks
- Extract only the spoken dialogue, not narrative descriptions
- Remove dialogue attribution tags (e.g., "he said", "she replied")
- If a character name is detected, keep it consistent with one speaker number`;
      }
    }

    // Add character mapping if available
    if (config.characterMapping && config.characterMapping.length > 0) {
      const mappingList = config.characterMapping
        .map((char) => `  - ${char.name} = Speaker ${char.speakerNumber}`)
        .join("\n");
      
      const hasNarrator = config.characterMapping.some(
        (char) => char.name.toLowerCase() === "narrator"
      );

      if (hasNarrator) {
        const narratorAttr = (config as any).narratorAttribution || 'remove';
        prompt += `\n\nCharacter to Speaker Mapping (use these exact assignments):
${mappingList}

IMPORTANT: 
- Only extract dialogue from the speaking characters listed above
- Assign narrative descriptions and non-dialogue portions to the Narrator
- ${narratorAttr === 'remove' ? 'Remove dialogue attribution tags like "he said", "she whispered", "Alice asked"' : narratorAttr === 'verbatim' ? 'Move dialogue attribution tags like "he said", "she whispered", "Alice asked" into a separate Narrator line placed immediately after the spoken line' : 'Transform dialogue attribution and actions into a separate Narrator line that summarizes the action (e.g., "John passed the book over to Tim.") placed immediately after the spoken line'}
- Ignore dialogue from any characters not in this mapping`;
      } else {
        prompt += `\n\nCharacter to Speaker Mapping (ONLY extract dialogue from these characters, use these exact assignments):
${mappingList}

IMPORTANT: Only extract dialogue from the characters listed above. Ignore dialogue from any other characters not in this mapping.`;
      }
    }

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    prompt += `\n\nText to format:\n${text}\n\nFormatted dialogue:`;

    return prompt;
  }

  private async runInference(prompt: string, options: ProcessChunkOptions): Promise<string> {
    const { modelSource = 'api', modelName, localModelName, ollamaModelName } = options;
    const resolvedModelName = normalizeHuggingFaceModelId(modelName);

    if (modelSource === 'ollama') {
      const model = ollamaModelName || "llama3.1:8b";
      const url = `${OLLAMA_BASE_URL}/api/generate`;
      const init: RequestInit = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
          options: {
            temperature: 0.3,
            num_predict: 2000,
          },
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
      return json.response || '';
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
          return response.choices?.[0]?.message?.content || '';
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
            return response.generated_text || '';
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
              return response.choices?.[0]?.message?.content || '';
            }
            throw e;
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
  }

  private buildSinglePassPrompt(
    text: string,
    cleaning: CleaningOptions,
    config: SpeakerConfig,
    customInstructions?: string
  ): string {
    const labelExample =
      config.labelFormat === "speaker"
        ? "Speaker 1:, Speaker 2:, ..."
        : "[1]:, [2]:, [3]:, ...";

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

    if (config.mode === 'format') {
      parts.push(`FORMAT: Convert to multi-speaker format. Labels: ${labelExample}. Only change labels; keep content.`);
    } else {
      if (hasNarrator) {
        const attrRule = narratorAttr === 'remove'
          ? 'Remove attribution tags (e.g., "he said").'
          : narratorAttr === 'verbatim'
            ? 'Move attribution tags into a Narrator line immediately after the spoken line.'
            : 'Convert attribution/action into a concise Narrator line after the spoken line (e.g., "John passed the book to Tim."). Prefer explicit names when ambiguous.';
        parts.push(`INTELLIGENT DIALOGUE: Detect speakers; output ${labelExample} for spoken lines. ${attrRule} Preserve narrative descriptions as Narrator lines. Use consistent speaker numbers.`);
      } else {
        parts.push(`INTELLIGENT DIALOGUE: Detect speakers; output ${labelExample} for spoken lines only. Remove attribution tags; keep consistent speaker numbering.`);
      }
    }

    if (config.characterMapping && config.characterMapping.length > 0) {
      const mappingList = config.characterMapping.map((c) => `${c.name} = Speaker ${c.speakerNumber}`).join('; ');
      parts.push(`MAPPING: ${mappingList}. Use exactly these assignments.`);
    }

    if (customInstructions) parts.push(`CUSTOM: ${customInstructions}`);
    parts.push(`Return ONLY the cleaned, formatted output. No explanations.`);

    return `${parts.join('\n')}
\nTEXT:\n${text}\n\nOUTPUT:`;
  }

  async processChunk(options: ProcessChunkOptions): Promise<string> {
    const { text, cleaningOptions, speakerConfig, customInstructions } = options;

    // Stage 1: Text cleaning
    let processedText = '';
    if (speakerConfig && speakerConfig.mode !== "none" && options.singlePass) {
      if (DEBUG_ENABLED) {
        console.debug('[LLM DEBUG] Single-pass processing enabled');
      }
      const singlePrompt = this.buildSinglePassPrompt(text, cleaningOptions, speakerConfig, customInstructions);
      processedText = await this.runInference(singlePrompt, options);
    } else {
      const useConcise = options.concisePrompts !== false;
      const cleaningPrompt = useConcise
        ? this.buildCleaningPromptConcise(text, cleaningOptions, customInstructions)
        : this.buildCleaningPrompt(text, cleaningOptions, customInstructions);
      processedText = await this.runInference(cleaningPrompt, options);
    }

    if (!processedText) {
      processedText = text;
    }

    // Stage 2: Speaker formatting (if configured and mode is not "none")
    if (speakerConfig && speakerConfig.mode !== "none" && !options.singlePass) {
      const speakerPrompt = this.buildSpeakerPrompt(processedText, speakerConfig, customInstructions);
      const stage2Text = await this.runInference(speakerPrompt, options);
      
      if (stage2Text) {
        processedText = stage2Text;
      }
    }

    return processedText.trim();
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
    concisePrompts?: boolean
  ): { stage1: string; stage2?: string } {
    const stage1 = speakerConfig && speakerConfig.mode !== 'none' && singlePass
      ? this.buildSinglePassPrompt(sampleText, cleaningOptions, speakerConfig, customInstructions)
      : (concisePrompts !== false
          ? this.buildCleaningPromptConcise(sampleText, cleaningOptions, customInstructions)
          : this.buildCleaningPrompt(sampleText, cleaningOptions, customInstructions));
    
    const result: { stage1: string; stage2?: string } = { stage1 };

    if (speakerConfig && speakerConfig.mode !== "none" && !singlePass) {
      result.stage2 = this.buildSpeakerPrompt(sampleText, speakerConfig, customInstructions);
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
  }): Promise<Array<{ name: string; speakerNumber: number }>> {
    const { text, includeNarrator, modelSource = 'api', modelName, localModelName, ollamaModelName } = options;
    const resolvedModelName = normalizeHuggingFaceModelId(modelName);

    const prompt = `You are a character extraction assistant for multi-speaker TTS systems. Analyze the following text sample and extract all character/speaker names that appear.

Requirements:
- Extract only actual character names that speak in the text
- List each unique character only once
- Return names in order of first appearance
${includeNarrator ? "- Include 'Narrator' as a character if there is narrative/descriptive text" : "- Do NOT include narrator or descriptive text"}
- Return ONLY a JSON array of character names, no explanations

Format your response as a JSON array:
["Character1", "Character2", "Character3"]

Text sample:
${text}

Character names (JSON array only):`;

    let content: string;

    if (modelSource === 'ollama') {
      const model = ollamaModelName || "llama3.1:8b";
      const url = `${OLLAMA_BASE_URL}/api/generate`;
      const init: RequestInit = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt, stream: false, options: { temperature: 0.2, num_predict: 500 } }),
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
      content = json.response || "[]";
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
    
    // Extract JSON array from response
    let characterNames: string[] = [];
    try {
      // Try to parse directly
      characterNames = JSON.parse(content.trim());
    } catch {
      // Try to extract JSON array from markdown code block or other formatting
      const jsonMatch = content.match(/\[[\s\S]*?\]/);
      if (jsonMatch) {
        try {
          characterNames = JSON.parse(jsonMatch[0]);
        } catch {
          console.error("Failed to parse character names from LLM response:", content);
        }
      }
    }

    // Ensure we have an array
    if (!Array.isArray(characterNames)) {
      characterNames = [];
    }

    // Map character names to speaker numbers (1-indexed)
    return characterNames.map((name, index) => ({
      name: String(name).trim(),
      speakerNumber: index + 1,
    }));
  }
}

export const llmService = new LLMService();
