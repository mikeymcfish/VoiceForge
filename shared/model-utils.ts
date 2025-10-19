import { ModelSource } from "./schema";

const THINKING_PATTERNS = [
  /thinking/, // generic flag
  /think\b/,
  /deepseek-?r1/,
  /reason/,
  /ponder/,
  /cogitat/,
];

function normalizeModelName(name?: string | null): string {
  return (name || "").trim().toLowerCase();
}

export function isThinkingOllamaModel(modelName?: string | null): boolean {
  const normalized = normalizeModelName(modelName);
  if (!normalized) return false;
  return THINKING_PATTERNS.some((pattern) => pattern.test(normalized));
}

const DEFAULT_CHARACTER_SAMPLE_MAX = 100;
const THINKING_CHARACTER_SAMPLE_MAX = 160;

export function getCharacterSampleCeiling(
  modelSource: ModelSource | undefined,
  ollamaModelName?: string | null
): number {
  if (modelSource === "ollama" && isThinkingOllamaModel(ollamaModelName)) {
    return THINKING_CHARACTER_SAMPLE_MAX;
  }
  return DEFAULT_CHARACTER_SAMPLE_MAX;
}

export function clampCharacterSampleSize(
  requested: number,
  modelSource: ModelSource | undefined,
  ollamaModelName?: string | null
): number {
  const ceiling = getCharacterSampleCeiling(modelSource, ollamaModelName);
  if (!Number.isFinite(requested)) return ceiling;
  return Math.max(5, Math.min(ceiling, Math.round(requested)));
}

export type OllamaGenerationOptions = {
  temperature?: number;
  num_predict?: number;
  num_ctx?: number;
  top_p?: number;
  repeat_penalty?: number;
};

export function buildOllamaOptions(
  base: OllamaGenerationOptions,
  modelSource: ModelSource | undefined,
  ollamaModelName?: string | null,
  overrides?: Partial<OllamaGenerationOptions>
): OllamaGenerationOptions {
  const isThinking = modelSource === "ollama" && isThinkingOllamaModel(ollamaModelName);
  const thinkingTuned: OllamaGenerationOptions = isThinking
    ? {
        temperature: 0.2,
        num_predict: Math.max(2048, base.num_predict ?? 0, 4096),
        num_ctx: Math.max(8192, base.num_ctx ?? 0),
        top_p: 0.9,
        repeat_penalty: 1.08,
      }
    : {};

  return Object.fromEntries(
    Object.entries({ ...base, ...thinkingTuned, ...overrides }).filter(([, value]) =>
      typeof value === "number" && Number.isFinite(value) ? true : value !== undefined
    )
  ) as OllamaGenerationOptions;
}
