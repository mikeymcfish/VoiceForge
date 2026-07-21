export type BatchProfileId = "compact" | "standard" | "large" | "extra-large";

export interface BatchProfile {
  id: BatchProfileId;
  label: string;
  description: string;
  batchSize: number;
  ollamaContextWindow: number;
  ollamaMaxOutputTokens: number;
}

/**
 * Local-inference profiles for text repair. Context and output limits are sent
 * only to Ollama; sentence batching applies to every provider.
 */
export const BATCH_PROFILES: readonly BatchProfile[] = [
  {
    id: "compact",
    label: "Compact",
    description: "25 sentences · 8K context · 2K response",
    batchSize: 25,
    ollamaContextWindow: 8_192,
    ollamaMaxOutputTokens: 2_000,
  },
  {
    id: "standard",
    label: "Standard",
    description: "50 sentences · 16K context · 4K response",
    batchSize: 50,
    ollamaContextWindow: 16_384,
    ollamaMaxOutputTokens: 4_096,
  },
  {
    id: "large",
    label: "Large",
    description: "120 sentences · 32K context · 8K response",
    batchSize: 120,
    ollamaContextWindow: 32_768,
    ollamaMaxOutputTokens: 8_192,
  },
  {
    id: "extra-large",
    label: "Extra large",
    description: "150 sentences · 48K context · 12K response",
    batchSize: 150,
    ollamaContextWindow: 49_152,
    ollamaMaxOutputTokens: 12_288,
  },
] as const;

export const MAX_BATCH_SIZE = Math.max(...BATCH_PROFILES.map((profile) => profile.batchSize));

export function getBatchProfileId(
  batchSize: number,
  ollamaContextWindow: number,
  ollamaMaxOutputTokens: number
): BatchProfileId | "custom" {
  return BATCH_PROFILES.find(
    (profile) =>
      profile.batchSize === batchSize &&
      profile.ollamaContextWindow === ollamaContextWindow &&
      profile.ollamaMaxOutputTokens === ollamaMaxOutputTokens
  )?.id ?? "custom";
}
