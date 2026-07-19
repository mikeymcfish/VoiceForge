export const VOICEFORGE_MODEL_IDS = [
  "index-tts-2",
  "vibevoice-1.5b",
  "vibevoice-large",
  "qwen3-tts-0.6b",
  "qwen3-tts-1.7b",
  "moss-tts-v1.5",
] as const;

export type VoiceForgeModelId = (typeof VOICEFORGE_MODEL_IDS)[number];
export type VoiceForgeTarget = "local" | "agent";
export type VoiceForgeTargetPreference = VoiceForgeTarget | "auto";
export type VoiceForgePreference = "speed" | "quality";
export type VoiceForgeMode =
  | "auto"
  | "direct"
  | "clone"
  | "design"
  | "preset"
  | "continuation"
  | "continuation-clone";

export type VoiceForgeModelStatus = {
  id: VoiceForgeModelId;
  label: string;
  repositoryId: string;
  targets: VoiceForgeTarget[];
  localModes: VoiceForgeMode[];
  agentModes: VoiceForgeMode[];
  localCharacterLimit: number;
  agentCharacterLimit?: number;
  localReady: boolean;
  agentReady: boolean;
  setupHint?: string;
};

export type SpeechRecommendationInput = {
  characterCount: number;
  target?: VoiceForgeTargetPreference;
  hasVoice?: boolean;
  speakerCount?: number;
  mode?: VoiceForgeMode;
  preference?: VoiceForgePreference;
  needsInlinePauses?: boolean;
};

export type SpeechRecommendationCandidate = {
  model: VoiceForgeModelId;
  target: VoiceForgeTarget;
  mode: Exclude<VoiceForgeMode, "auto">;
  runnableNow: boolean;
  blockers: string[];
};

export type SpeechRecommendation = {
  characterCount: number;
  lengthBand: "empty" | "short" | "medium" | "long" | "too-long";
  requestedTarget: VoiceForgeTargetPreference;
  recommended?: SpeechRecommendationCandidate;
  alternatives: SpeechRecommendationCandidate[];
  reasons: string[];
  warnings: string[];
};

export const MODEL_REPOSITORIES: Record<VoiceForgeModelId, string> = {
  "index-tts-2": "IndexTeam/IndexTTS-2",
  "vibevoice-1.5b": "microsoft/VibeVoice-1.5B",
  "vibevoice-large": "aoi-ot/VibeVoice-Large",
  "qwen3-tts-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "qwen3-tts-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "moss-tts-v1.5": "OpenMOSS-Team/MOSS-TTS-v1.5",
};

const MAX_LOCAL_CHARACTERS = 500_000;
const QWEN_AGENT_LIMIT = 1_200;
const MOSS_AGENT_LIMIT = 5_000;

function lengthBand(characterCount: number): SpeechRecommendation["lengthBand"] {
  if (characterCount <= 0) return "empty";
  if (characterCount <= QWEN_AGENT_LIMIT) return "short";
  if (characterCount <= MOSS_AGENT_LIMIT) return "medium";
  if (characterCount <= MAX_LOCAL_CHARACTERS) return "long";
  return "too-long";
}

function statusFor(
  statuses: readonly VoiceForgeModelStatus[],
  model: VoiceForgeModelId
): VoiceForgeModelStatus | undefined {
  return statuses.find((status) => status.id === model);
}

export function resolveVoiceForgeMode(
  model: VoiceForgeModelId,
  target: VoiceForgeTarget,
  requestedMode: VoiceForgeMode,
  hasVoice: boolean
): Exclude<VoiceForgeMode, "auto"> {
  if (requestedMode !== "auto") return requestedMode;
  if (model === "moss-tts-v1.5") return hasVoice ? "clone" : "direct";
  if (model.startsWith("qwen3-tts") && target === "agent" && !hasVoice) {
    return model === "qwen3-tts-0.6b" ? "preset" : "design";
  }
  return "clone";
}

function candidate(
  statuses: readonly VoiceForgeModelStatus[],
  model: VoiceForgeModelId,
  target: VoiceForgeTarget,
  mode: Exclude<VoiceForgeMode, "auto">,
  characterCount: number,
  hasVoice: boolean,
  speakerCount: number
): SpeechRecommendationCandidate | undefined {
  const status = statusFor(statuses, model);
  if (!status || !status.targets.includes(target)) return undefined;
  const modes = target === "local" ? status.localModes : status.agentModes;
  if (!modes.includes(mode)) return undefined;
  if (characterCount > (target === "local" ? status.localCharacterLimit : status.agentCharacterLimit ?? 0)) {
    return undefined;
  }

  const blockers: string[] = [];
  const needsVoice =
    model === "index-tts-2" ||
    model === "vibevoice-1.5b" ||
    model === "vibevoice-large" ||
    model.startsWith("qwen3-tts") && mode === "clone" ||
    model === "moss-tts-v1.5" && mode !== "direct";
  if (needsVoice && !hasVoice) blockers.push("A reference voice is required for this mode.");
  if ((model === "vibevoice-1.5b" || model === "vibevoice-large") && speakerCount > 4) {
    blockers.push("VibeVoice supports at most four reference voices.");
  }
  const ready = target === "local" ? status.localReady : status.agentReady;
  if (!ready) blockers.push(status.setupHint || `${status.label} is not ready for ${target} inference.`);
  return { model, target, mode, runnableNow: blockers.length === 0, blockers };
}

export function voiceForgeModelSupportsAudioProcessing(model: VoiceForgeModelId): boolean {
  return model.startsWith("qwen3-tts") || model === "moss-tts-v1.5";
}

export function recommendAudioProcessingSpeechModel(
  input: SpeechRecommendationInput,
  statuses: readonly VoiceForgeModelStatus[]
): SpeechRecommendation {
  return recommendSpeechModel(
    input,
    statuses.filter((status) => voiceForgeModelSupportsAudioProcessing(status.id))
  );
}

export function recommendSpeechModel(
  input: SpeechRecommendationInput,
  statuses: readonly VoiceForgeModelStatus[]
): SpeechRecommendation {
  const characterCount = Math.max(0, Math.trunc(input.characterCount));
  const band = lengthBand(characterCount);
  const target = input.target ?? "auto";
  const hasVoice = input.hasVoice === true;
  const speakerCount = Math.max(1, Math.trunc(input.speakerCount ?? (hasVoice ? 1 : 1)));
  const requestedMode = input.mode ?? "auto";
  const preference = input.preference ?? "quality";
  const warnings: string[] = [];
  const reasons: string[] = [];

  if (band === "empty") {
    return {
      characterCount,
      lengthBand: band,
      requestedTarget: target,
      alternatives: [],
      reasons: [],
      warnings: ["Text is required before a speech model can be recommended."],
    };
  }
  if (band === "too-long") {
    return {
      characterCount,
      lengthBand: band,
      requestedTarget: target,
      alternatives: [],
      reasons: [],
      warnings: ["VoiceForge accepts at most 500,000 characters per local job; split the text first."],
    };
  }
  if (speakerCount > 4) {
    return {
      characterCount,
      lengthBand: band,
      requestedTarget: target,
      alternatives: [],
      reasons: [],
      warnings: ["VoiceForge supports at most four mapped speakers in one multi-speaker job."],
    };
  }

  const ranked: Array<{ model: VoiceForgeModelId; target: VoiceForgeTarget }> = [];
  const add = (model: VoiceForgeModelId, candidateTarget: VoiceForgeTarget) => {
    if (target !== "auto" && target !== candidateTarget) return;
    if (!ranked.some((item) => item.model === model && item.target === candidateTarget)) {
      ranked.push({ model, target: candidateTarget });
    }
  };

  if (speakerCount > 1) {
    add(preference === "speed" ? "vibevoice-1.5b" : "vibevoice-large", "local");
    add(preference === "speed" ? "vibevoice-large" : "vibevoice-1.5b", "local");
    reasons.push("VibeVoice is the only VoiceForge engine with up to four mapped voice roles.");
  } else if (input.needsInlinePauses || requestedMode === "continuation" || requestedMode === "continuation-clone") {
    if (characterCount <= MOSS_AGENT_LIMIT) add("moss-tts-v1.5", "agent");
    add("moss-tts-v1.5", "local");
    reasons.push("MOSS supports continuation modes and inline [pause …] controls.");
  } else if (!hasVoice && requestedMode === "design" && characterCount <= QWEN_AGENT_LIMIT) {
    add("qwen3-tts-1.7b", "agent");
    reasons.push("Qwen Agent voice-design mode uses the Space's dedicated 1.7B voice-design model.");
  } else if (!hasVoice && requestedMode === "preset" && characterCount <= QWEN_AGENT_LIMIT) {
    add(preference === "speed" ? "qwen3-tts-0.6b" : "qwen3-tts-1.7b", "agent");
    add(preference === "speed" ? "qwen3-tts-1.7b" : "qwen3-tts-0.6b", "agent");
    reasons.push("Qwen Agent preset mode uses a built-in speaker and does not require a reference recording.");
  } else if (!hasVoice) {
    add("moss-tts-v1.5", "local");
    if (characterCount <= MOSS_AGENT_LIMIT) add("moss-tts-v1.5", "agent");
    reasons.push("MOSS direct generation does not require a reference voice.");
  } else if (target === "agent") {
    if (characterCount <= QWEN_AGENT_LIMIT) {
      add(preference === "speed" ? "qwen3-tts-0.6b" : "qwen3-tts-1.7b", "agent");
      add(preference === "speed" ? "qwen3-tts-1.7b" : "qwen3-tts-0.6b", "agent");
      reasons.push("Qwen is the shorter Agent path for text up to 1,200 characters.");
    } else if (characterCount <= MOSS_AGENT_LIMIT) {
      add("moss-tts-v1.5", "agent");
      reasons.push("MOSS Agent accepts text up to 5,000 characters per request.");
    }
  } else if (band === "long") {
    add(preference === "speed" ? "vibevoice-1.5b" : "vibevoice-large", "local");
    add("moss-tts-v1.5", "local");
    add("qwen3-tts-1.7b", "local");
    reasons.push("Long text must run locally; VibeVoice chunks long-form scripts.");
  } else {
    add(preference === "speed" ? "qwen3-tts-0.6b" : "qwen3-tts-1.7b", "local");
    add(preference === "speed" ? "qwen3-tts-1.7b" : "qwen3-tts-0.6b", "local");
    add("moss-tts-v1.5", "local");
    add("index-tts-2", "local");
    if (target === "auto") {
      if (characterCount <= QWEN_AGENT_LIMIT) {
        add(preference === "speed" ? "qwen3-tts-0.6b" : "qwen3-tts-1.7b", "agent");
      } else if (characterCount <= MOSS_AGENT_LIMIT) {
        add("moss-tts-v1.5", "agent");
      }
    }
    reasons.push("Qwen is the compact local cloning path for short and medium single-voice text.");
  }

  if (target === "agent" && characterCount > MOSS_AGENT_LIMIT) {
    warnings.push("Agent inference is limited to 5,000 characters; choose Local rather than silently splitting the request.");
  }

  const candidates = ranked
    .map(({ model, target: candidateTarget }) =>
      candidate(
        statuses,
        model,
        candidateTarget,
        resolveVoiceForgeMode(model, candidateTarget, requestedMode, hasVoice),
        characterCount,
        hasVoice,
        speakerCount
      )
    )
    .filter((item): item is SpeechRecommendationCandidate => Boolean(item));
  const runnable = candidates.filter((item) => item.runnableNow);
  const recommended = runnable[0] ?? candidates[0];
  const alternatives = candidates.filter((item) => item !== recommended).slice(0, 4);
  if (recommended && !recommended.runnableNow) {
    warnings.push("The best-fit model is not ready yet; review its blockers before generating.");
  }
  return {
    characterCount,
    lengthBand: band,
    requestedTarget: target,
    recommended,
    alternatives,
    reasons,
    warnings,
  };
}
