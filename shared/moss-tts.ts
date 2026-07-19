export const MOSS_DURATION_TOKENS_PLACEHOLDER = 1;

export const MOSS_DELAY_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5";
export const MOSS_LOCAL_MODEL_ID =
  "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5";

export const MOSS_LOCAL_CHECKPOINTS = [
  {
    id: MOSS_DELAY_MODEL_ID,
    label: "MOSS-TTS v1.5 8B (production default)",
    description: "Best long-context stability and production-quality cloning.",
  },
  {
    id: MOSS_LOCAL_MODEL_ID,
    label: "Local-Transformer v1.5 5B (lower VRAM)",
    description: "Smaller streaming-oriented checkpoint with 48 kHz stereo output.",
  },
] as const;

/**
 * The official MOSS Space exposes duration_tokens as a hidden 1-2 slider while
 * duration control is disabled. Keep its placeholder value out of the normal
 * duration estimate so a stale UI value cannot fail Gradio input validation.
 */
export function mossHostedDurationTokens(
  enabled: boolean,
  requested?: number
): number {
  return enabled && requested !== undefined
    ? requested
    : MOSS_DURATION_TOKENS_PLACEHOLDER;
}
