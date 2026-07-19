import { createHash } from "node:crypto";
import fs from "node:fs";
import fsPromises from "node:fs/promises";
import type {
  SpeechRecommendation,
  SpeechRecommendationInput,
  VoiceForgeMode,
  VoiceForgeModelId,
  VoiceForgeModelStatus,
  VoiceForgePreference,
  VoiceForgeTarget,
  VoiceForgeTargetPreference,
} from "@shared/speech-recommendation";
import {
  MODEL_REPOSITORIES,
  recommendAudioProcessingSpeechModel,
  recommendSpeechModel,
  resolveVoiceForgeMode,
  voiceForgeModelSupportsAudioProcessing,
} from "@shared/speech-recommendation";
import type {
  SpeechOutputFormat,
  SpeechReferenceEnhancement,
} from "@shared/schema";
import { indexTtsService } from "./tts-service";
import { vibevoiceService } from "./vibevoice-service";
import { speechService } from "./speech-service";
import {
  voiceLibraryService,
  type DefaultVoiceMetadata,
  type ReadDefaultVoice,
} from "./voice-library-service";

type PublicJobState = "queued" | "running" | "completed" | "failed" | "cancelled";
type JobEngine = "index" | "vibe" | "qwen" | "moss";

export type VoiceForgeJob = {
  id: string;
  model: VoiceForgeModelId;
  target: VoiceForgeTarget;
  mode: Exclude<VoiceForgeMode, "auto">;
  status: PublicJobState;
  progress: number;
  message?: string;
  error?: string;
  createdAt: number;
  updatedAt: number;
  outputFormat: SpeechOutputFormat;
  outputMimeType: "audio/wav" | "audio/mpeg";
  chapterCount: number;
  referenceEnhancement: SpeechReferenceEnhancement;
  levelNormalized?: boolean;
  audioResourceUri?: string;
  audioPath?: string;
};

export type GenerateSpeechInput = {
  requestId: string;
  text: string;
  model: VoiceForgeModelId | "auto";
  target: VoiceForgeTarget;
  mode?: VoiceForgeMode;
  voiceId?: string;
  voiceIds?: string[];
  language?: string;
  referenceText?: string;
  voiceDescription?: string;
  speaker?: string;
  instruction?: string;
  preference?: VoiceForgePreference;
  style?: string;
  guidanceScale?: number;
  needsInlinePauses?: boolean;
  outputFormat?: SpeechOutputFormat;
  useChapters?: boolean;
  chapterPauseMs?: number;
  mp3Quality?: number;
  normalizeLevels?: boolean;
  referenceEnhancement?: SpeechReferenceEnhancement;
  audioSrModel?: "speech" | "basic";
  audioSrDevice?: string;
  audioSrDdimSteps?: number;
  audioSrGuidanceScale?: number;
  audioSrSeed?: number;
};

export type GenerateSpeechResult = {
  job: VoiceForgeJob;
  selectedByRecommendation: boolean;
  recommendationReasons: string[];
  warnings: string[];
};

type IdempotencyEntry = {
  inputHash: string;
  jobId: string;
  createdAt: number;
  reasons: string[];
  warnings: string[];
  selectedByRecommendation: boolean;
};

type InFlightEntry = {
  inputHash: string;
  promise: Promise<GenerateSpeechResult>;
};

const MAX_TEXT_CHARACTERS = 500_000;
const IDEMPOTENCY_TTL_MS = 24 * 60 * 60 * 1_000;
const MODEL_LABELS: Record<VoiceForgeModelId, string> = {
  "index-tts-2": "IndexTTS2",
  "vibevoice-1.5b": "VibeVoice 1.5B",
  "vibevoice-large": "VibeVoice Large",
  "qwen3-tts-0.6b": "Qwen3-TTS 0.6B Base",
  "qwen3-tts-1.7b": "Qwen3-TTS 1.7B Base",
  "moss-tts-v1.5": "MOSS-TTS v1.5",
};

function externalJobId(engine: JobEngine, id: string): string {
  return `vf-${engine}-${id}`;
}

function parseJobId(id: string): { engine: JobEngine; id: string } | undefined {
  const match = /^vf-(index|vibe|qwen|moss)-([A-Za-z0-9_-]+)$/.exec(id);
  if (!match) return undefined;
  return { engine: match[1] as JobEngine, id: match[2] };
}

function modelForEngine(engine: JobEngine, selectedModel?: string): VoiceForgeModelId {
  if (engine === "index") return "index-tts-2";
  if (engine === "moss") return "moss-tts-v1.5";
  if (engine === "qwen") return selectedModel?.includes("0.6B") ? "qwen3-tts-0.6b" : "qwen3-tts-1.7b";
  return selectedModel === MODEL_REPOSITORIES["vibevoice-large"]
    ? "vibevoice-large"
    : "vibevoice-1.5b";
}

function sanitizeLanguage(value?: string): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) return undefined;
  if (trimmed.length > 40 || /[\u0000-\u001f\u007f]/u.test(trimmed)) {
    throw new Error("Language must be a short plain-text label.");
  }
  return trimmed;
}

function hashInput(input: GenerateSpeechInput): string {
  return createHash("sha256").update(JSON.stringify(input)).digest("hex");
}

function isVibeModel(model: VoiceForgeModelId): boolean {
  return model === "vibevoice-1.5b" || model === "vibevoice-large";
}

function finiteNumberInRange(
  label: string,
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  integer = false
): number {
  const result = value ?? fallback;
  if (!Number.isFinite(result) || result < minimum || result > maximum || (integer && !Number.isInteger(result))) {
    throw new Error(
      `${label} must be ${integer ? "an integer" : "a number"} from ${minimum.toLocaleString()} to ${maximum.toLocaleString()}.`
    );
  }
  return result;
}

export class VoiceForgeOrchestrator {
  private readonly idempotency = new Map<string, IdempotencyEntry>();
  private readonly inFlight = new Map<string, InFlightEntry>();

  public getModelStatuses(): VoiceForgeModelStatus[] {
    const index = indexTtsService.getStatus();
    const vibe = vibevoiceService.getStatus();
    const speech = speechService.getStatus();
    const qwen = speech.engines.find((engine) => engine.engine === "qwen");
    const moss = speech.engines.find((engine) => engine.engine === "moss");
    const indexReady = index.runtimeConfigured && index.modelsReady && index.loadStatus === "completed";

    const make = (
      id: VoiceForgeModelId,
      targets: VoiceForgeTarget[],
      localModes: VoiceForgeMode[],
      agentModes: VoiceForgeMode[],
      localReady: boolean,
      agentReady: boolean,
      setupHint: string,
      agentCharacterLimit?: number
    ): VoiceForgeModelStatus => ({
      id,
      label: MODEL_LABELS[id],
      repositoryId: MODEL_REPOSITORIES[id],
      targets,
      localModes,
      agentModes,
      localCharacterLimit: MAX_TEXT_CHARACTERS,
      agentCharacterLimit,
      localReady,
      agentReady,
      setupHint,
    });

    return [
      make(
        "index-tts-2",
        ["local"],
        ["clone"],
        [],
        indexReady,
        false,
        "Run VoiceForge.cmd setup-index, download the pinned model, and verify the runtime."
      ),
      make(
        "vibevoice-1.5b",
        ["local"],
        ["clone"],
        [],
        vibe.ready && vibe.availableModels.some((model) => model.id === MODEL_REPOSITORIES["vibevoice-1.5b"]),
        false,
        "Install the pinned VibeVoice runtime and 1.5B model in Create audio."
      ),
      make(
        "vibevoice-large",
        ["local"],
        ["clone"],
        [],
        vibe.ready && vibe.availableModels.some((model) => model.id === MODEL_REPOSITORIES["vibevoice-large"]),
        false,
        "Install the pinned VibeVoice runtime and Large model in Create audio."
      ),
      make(
        "qwen3-tts-0.6b",
        ["local", "agent"],
        ["clone"],
        ["clone", "preset"],
        Boolean(qwen?.runtimeConfigured && qwen.availableModels.includes(MODEL_REPOSITORIES["qwen3-tts-0.6b"])),
        Boolean(speech.tokenConfigured && qwen?.hostedAvailable),
        "Run VoiceForge.cmd setup-qwen for Local, or configure a standard Hugging Face read token for Agent.",
        1_200
      ),
      make(
        "qwen3-tts-1.7b",
        ["local", "agent"],
        ["clone"],
        ["clone", "design", "preset"],
        Boolean(qwen?.runtimeConfigured && qwen.availableModels.includes(MODEL_REPOSITORIES["qwen3-tts-1.7b"])),
        Boolean(speech.tokenConfigured && qwen?.hostedAvailable),
        "Run VoiceForge.cmd setup-qwen for Local, or configure a standard Hugging Face read token for Agent.",
        1_200
      ),
      make(
        "moss-tts-v1.5",
        ["local", "agent"],
        ["direct", "clone"],
        ["direct", "clone", "continuation", "continuation-clone"],
        Boolean(moss?.runtimeConfigured && moss.availableModels.includes(MODEL_REPOSITORIES["moss-tts-v1.5"])),
        Boolean(speech.tokenConfigured && moss?.hostedAvailable),
        "Run VoiceForge.cmd setup-moss for Local, or configure a standard Hugging Face read token for Agent.",
        5_000
      ),
    ];
  }

  public async listVoices(): Promise<DefaultVoiceMetadata[]> {
    return voiceLibraryService.listVoices();
  }

  public async readVoiceAudio(id: string): Promise<{ audio: Buffer; mimeType: string; name: string } | undefined> {
    const voice = await voiceLibraryService.readVoice(id);
    return voice
      ? { audio: voice.audio, mimeType: voice.metadata.mimeType, name: voice.metadata.name }
      : undefined;
  }

  public recommend(input: SpeechRecommendationInput): SpeechRecommendation {
    return recommendSpeechModel(input, this.getModelStatuses());
  }

  private cleanupIdempotency(): void {
    const cutoff = Date.now() - IDEMPOTENCY_TTL_MS;
    for (const [requestId, entry] of this.idempotency) {
      if (entry.createdAt < cutoff) this.idempotency.delete(requestId);
    }
    while (this.idempotency.size > 500) {
      const oldest = this.idempotency.keys().next().value as string | undefined;
      if (!oldest) break;
      this.idempotency.delete(oldest);
    }
  }

  private async readVoices(ids: readonly string[]): Promise<ReadDefaultVoice[]> {
    const voices: ReadDefaultVoice[] = [];
    for (const id of ids) {
      const voice = await voiceLibraryService.readVoice(id);
      if (!voice) throw new Error(`Default voice ${id} was not found or is no longer safe to read.`);
      voices.push(voice);
    }
    return voices;
  }

  public async generate(input: GenerateSpeechInput): Promise<GenerateSpeechResult> {
    const text = input.text.trim();
    if (!text) throw new Error("Text is required.");
    if (text.length > MAX_TEXT_CHARACTERS) {
      throw new Error("VoiceForge accepts at most 500,000 characters per local job.");
    }
    if (!/^[A-Za-z0-9_-]{8,128}$/.test(input.requestId)) {
      throw new Error("requestId must be 8-128 URL-safe characters so retries remain idempotent.");
    }
    const rawVoiceIds = input.voiceIds?.length ? input.voiceIds : input.voiceId ? [input.voiceId] : [];
    if (rawVoiceIds.length > 4) throw new Error("At most four default voices can be selected.");
    const voices = await this.readVoices(rawVoiceIds);
    const hasVoice = voices.length > 0;
    const requestedMode = input.mode ?? "auto";

    this.cleanupIdempotency();
    const inputHash = hashInput({ ...input, text });
    const existing = this.idempotency.get(input.requestId);
    if (existing) {
      if (existing.inputHash !== inputHash) {
        throw new Error("This requestId was already used with different synthesis inputs.");
      }
      const job = this.getJob(existing.jobId);
      if (!job) throw new Error("The idempotent job is no longer available; use a new requestId.");
      return {
        job,
        selectedByRecommendation: existing.selectedByRecommendation,
        recommendationReasons: existing.reasons,
        warnings: existing.warnings,
      };
    }

    const pending = this.inFlight.get(input.requestId);
    if (pending) {
      if (pending.inputHash !== inputHash) {
        throw new Error("This requestId is already starting a job with different synthesis inputs.");
      }
      return pending.promise;
    }

    // Defer execution by one microtask so the in-flight reservation is visible
    // before any engine can start, preventing concurrent retries from creating
    // duplicate synthesis jobs.
    const operation = Promise.resolve().then(async (): Promise<GenerateSpeechResult> => {
      const audioProcessingRequested = [
        input.outputFormat,
        input.useChapters,
        input.chapterPauseMs,
        input.mp3Quality,
        input.normalizeLevels === true ? true : undefined,
        input.referenceEnhancement,
        input.audioSrModel,
        input.audioSrDevice,
        input.audioSrDdimSteps,
        input.audioSrGuidanceScale,
        input.audioSrSeed,
      ].some((value) => value !== undefined);
      let model = input.model;
      let recommendation: SpeechRecommendation | undefined;
      if (model === "auto") {
        const recommendationInput = {
          characterCount: text.length,
          target: input.target,
          hasVoice,
          speakerCount: Math.max(1, voices.length),
          mode: requestedMode,
          preference: input.preference,
          needsInlinePauses: input.needsInlinePauses,
        } satisfies SpeechRecommendationInput;
        recommendation = audioProcessingRequested
          ? recommendAudioProcessingSpeechModel(
              recommendationInput,
              this.getModelStatuses()
            )
          : this.recommend(recommendationInput);
        if (!recommendation.recommended) {
          throw new Error(recommendation.warnings[0] || "No compatible speech model is available.");
        }
        model = recommendation.recommended.model;
      }

      const status = this.getModelStatuses().find((item) => item.id === model);
      if (!status) throw new Error("Choose a supported VoiceForge model.");
      if (audioProcessingRequested && !voiceForgeModelSupportsAudioProcessing(model)) {
        throw new Error("Level normalization, MP3 export, chapters, and reference-audio enhancement are available only with Qwen3-TTS or MOSS-TTS.");
      }
      if (!status.targets.includes(input.target)) {
        throw new Error(`${status.label} does not support ${input.target === "agent" ? "Hugging Face Agent" : "Local"} inference.`);
      }
      const limit = input.target === "local" ? status.localCharacterLimit : status.agentCharacterLimit ?? 0;
      if (text.length > limit) {
        throw new Error(
          input.target === "agent"
            ? `${status.label} Agent requests are limited to ${limit.toLocaleString()} characters; choose Local.`
            : `${status.label} is limited to ${limit.toLocaleString()} characters per local job.`
        );
      }
      const mode = resolveVoiceForgeMode(model, input.target, requestedMode, hasVoice);
      const supportedModes = input.target === "local" ? status.localModes : status.agentModes;
      if (!supportedModes.includes(mode)) throw new Error(`${status.label} does not support ${mode} mode on this target.`);
      if ((input.target === "local" && !status.localReady) || (input.target === "agent" && !status.agentReady)) {
        throw new Error(status.setupHint || `${status.label} is not ready.`);
      }
      const needsVoice =
        model === "index-tts-2" || isVibeModel(model) ||
        (model.startsWith("qwen3-tts") && mode === "clone") ||
        (model === "moss-tts-v1.5" && mode !== "direct");
      if (needsVoice && !hasVoice) throw new Error(`${status.label} ${mode} mode requires a default voice ID.`);
      if (!isVibeModel(model) && voices.length > 1) throw new Error(`${status.label} accepts one reference voice per job.`);

      const outputFormat = input.outputFormat ?? "wav";
      const normalizeLevels = input.normalizeLevels ?? true;
      const useChapters = input.useChapters ?? false;
      if (useChapters && input.target !== "local") {
        throw new Error("Exact MP3 chapter timing is available for Local synthesis only.");
      }
      if (useChapters && outputFormat !== "mp3") {
        throw new Error("Set output_format to mp3 when enabling exact MP3 chapters.");
      }
      const chapterMarkerCount = (text.match(/\[chapter\]/giu) || []).length;
      const hasChapterContent = text
        .split(/\[chapter\]/iu)
        .slice(1)
        .some((section) => section.trim().length > 0);
      if (useChapters && (chapterMarkerCount === 0 || !hasChapterContent)) {
        throw new Error(
          "Add at least one [CHAPTER] marker followed by spoken text before enabling MP3 chapters."
        );
      }
      if (useChapters && chapterMarkerCount > 500) {
        throw new Error("A synthesis job may contain at most 500 [CHAPTER] markers.");
      }
      const referenceEnhancement = input.referenceEnhancement ?? "none";
      if (referenceEnhancement !== "none" && !needsVoice) {
        throw new Error(
          "Reference-audio enhancement is available only in a mode that uses a reference voice."
        );
      }
      if (referenceEnhancement !== "none" && !hasVoice) {
        throw new Error("Reference-audio enhancement requires a default voice ID.");
      }
      const chapterPauseMs = finiteNumberInRange("chapterPauseMs", input.chapterPauseMs, 0, 0, 10_000, true);
      const mp3Quality = finiteNumberInRange("mp3Quality", input.mp3Quality, 2, 0, 9, true);
      const audioSrDdimSteps = finiteNumberInRange("audioSrDdimSteps", input.audioSrDdimSteps, 50, 10, 250, true);
      const audioSrGuidanceScale = finiteNumberInRange(
        "audioSrGuidanceScale",
        input.audioSrGuidanceScale,
        3.5,
        1,
        10
      );
      const audioSrSeed = finiteNumberInRange(
        "audioSrSeed",
        input.audioSrSeed,
        42,
        -2_147_483_648,
        2_147_483_647,
        true
      );
      const audioSrDevice = input.audioSrDevice?.trim().toLowerCase() || "auto";
      if (!/^(?:auto|cpu|mps|cuda(?::\d{1,3})?)$/u.test(audioSrDevice)) {
        throw new Error(
          "audioSrDevice must be auto, cpu, mps, cuda, or cuda:N (up to three digits)."
        );
      }

      let rawJobId: string;
      let engine: JobEngine;
      if (model === "index-tts-2") {
      const voice = voices[0];
      const job = await indexTtsService.startSynthesis({
        voiceBuffer: voice.audio,
        voiceFileName: `${voice.metadata.name}.${voice.metadata.format}`,
        textContent: text,
        textFileName: "mcp-script.txt",
      });
      rawJobId = job.id;
      engine = "index";
      } else if (isVibeModel(model)) {
      const job = await vibevoiceService.startSynthesis({
        voiceBuffers: voices.map((voice) => voice.audio),
        voiceFileNames: voices.map((voice) => `${voice.metadata.name}.${voice.metadata.format}`),
        textContent: text,
        textFileName: "mcp-script.txt",
        style: input.style?.trim().slice(0, 500),
        guidanceScale: input.guidanceScale,
        modelId: MODEL_REPOSITORIES[model],
      });
      rawJobId = job.id;
      engine = "vibe";
      } else {
      const voice = voices[0];
      const internalMode = mode === "preset" ? "custom" : mode;
      const job = await speechService.startSynthesis({
        engine: model === "moss-tts-v1.5" ? "moss" : "qwen",
        target: input.target === "agent" ? "hf-space" : "local",
        mode: internalMode,
        text,
        voiceBuffer: voice?.audio,
        voiceFileName: voice ? `${voice.metadata.name}.${voice.metadata.format}` : undefined,
        modelId: MODEL_REPOSITORIES[model],
        referenceText: input.referenceText?.trim() || voice?.metadata.transcript || undefined,
        language: sanitizeLanguage(input.language),
        voiceDescription: input.voiceDescription?.trim().slice(0, 1_000),
        speaker: input.speaker,
        instruction: input.instruction?.trim().slice(0, 1_000),
        modelSize: model === "qwen3-tts-0.6b" ? "0.6B" : "1.7B",
        outputFormat,
        normalizeLevels,
        useChapters,
        chapterPauseMs,
        mp3Quality,
        referenceEnhancement,
        audioSrModel: input.audioSrModel ?? "speech",
        audioSrDevice,
        audioSrDdimSteps,
        audioSrGuidanceScale,
        audioSrSeed,
      });
      rawJobId = job.id;
      engine = model === "moss-tts-v1.5" ? "moss" : "qwen";
      }

      const jobId = externalJobId(engine, rawJobId);
      const entry: IdempotencyEntry = {
        inputHash,
        jobId,
        createdAt: Date.now(),
        reasons: recommendation?.reasons ?? [],
        warnings: recommendation?.warnings ?? [],
        selectedByRecommendation: input.model === "auto",
      };
      this.idempotency.set(input.requestId, entry);
      const job = this.getJob(jobId);
      if (!job) throw new Error("VoiceForge started the job but could not read its status.");
      return {
        job,
        selectedByRecommendation: entry.selectedByRecommendation,
        recommendationReasons: entry.reasons,
        warnings: entry.warnings,
      };
    });
    this.inFlight.set(input.requestId, { inputHash, promise: operation });
    try {
      return await operation;
    } finally {
      if (this.inFlight.get(input.requestId)?.promise === operation) {
        this.inFlight.delete(input.requestId);
      }
    }
  }

  public getJob(id: string): VoiceForgeJob | undefined {
    const parsed = parseJobId(id);
    if (!parsed) return undefined;
    if (parsed.engine === "index") {
      const job = indexTtsService.getJob(parsed.id);
      return job ? this.normalizeJob(id, "index-tts-2", "local", "clone", job) : undefined;
    }
    if (parsed.engine === "vibe") {
      const job = vibevoiceService.getJob(parsed.id);
      const model = modelForEngine("vibe", job?.selectedModel);
      return job ? this.normalizeJob(id, model, "local", "clone", job) : undefined;
    }
    const job = speechService.getJob(parsed.id);
    if (!job || job.engine !== parsed.engine) return undefined;
    const model = modelForEngine(parsed.engine, job.modelId);
    const mode = (job.mode === "custom" ? "preset" : job.mode) as Exclude<VoiceForgeMode, "auto">;
    return this.normalizeJob(id, model, job.target === "hf-space" ? "agent" : "local", mode, job);
  }

  private normalizeJob(
    id: string,
    model: VoiceForgeModelId,
    target: VoiceForgeTarget,
    mode: Exclude<VoiceForgeMode, "auto">,
    job: {
      status: PublicJobState;
      progress: number;
      message?: string;
      error?: string;
      createdAt: number;
      updatedAt: number;
      outputFormat?: SpeechOutputFormat;
      outputMimeType?: "audio/wav" | "audio/mpeg";
      chapterCount?: number;
      referenceEnhancement?: SpeechReferenceEnhancement;
      levelNormalized?: boolean;
    }
  ): VoiceForgeJob {
    const completed = job.status === "completed";
    const outputFormat = job.outputFormat ?? "wav";
    return {
      id,
      model,
      target,
      mode,
      status: job.status,
      progress: job.progress,
      message: job.message,
      error: job.error,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      outputFormat,
      outputMimeType: outputFormat === "mp3" ? "audio/mpeg" : "audio/wav",
      chapterCount: job.chapterCount ?? 0,
      referenceEnhancement: job.referenceEnhancement ?? "none",
      levelNormalized: job.levelNormalized ?? false,
      audioResourceUri: completed ? `voiceforge://speech/jobs/${id}/audio` : undefined,
      audioPath: completed ? `/api/mcp/speech/jobs/${id}/audio` : undefined,
    };
  }

  public cancelJob(id: string): VoiceForgeJob | undefined {
    const parsed = parseJobId(id);
    if (!parsed) return undefined;
    if (parsed.engine === "index") indexTtsService.cancelJob(parsed.id);
    else if (parsed.engine === "vibe") vibevoiceService.cancelJob(parsed.id);
    else {
      const job = speechService.getJob(parsed.id);
      if (!job || job.engine !== parsed.engine) return undefined;
      speechService.cancelJob(parsed.id);
    }
    return this.getJob(id);
  }

  public getJobOutputPath(id: string): string | undefined {
    const parsed = parseJobId(id);
    if (!parsed) return undefined;
    if (parsed.engine === "index") return indexTtsService.getJobOutputPath(parsed.id);
    if (parsed.engine === "vibe") return vibevoiceService.getJobOutputPath(parsed.id);
    const job = speechService.getJob(parsed.id);
    if (!job || job.engine !== parsed.engine) return undefined;
    return speechService.getJobOutputPath(parsed.id);
  }

  public async readJobAudio(id: string): Promise<Buffer | undefined> {
    const outputPath = this.getJobOutputPath(id);
    if (!outputPath || !fs.existsSync(outputPath)) return undefined;
    const stat = await fsPromises.stat(outputPath);
    if (!stat.isFile() || stat.size <= 0 || stat.size > 250 * 1024 * 1024) return undefined;
    return fsPromises.readFile(outputPath);
  }
}

export const voiceForgeOrchestrator = new VoiceForgeOrchestrator();
