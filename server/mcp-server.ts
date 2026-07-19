import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  VOICEFORGE_MODEL_IDS,
  type SpeechRecommendation,
  type VoiceForgeMode,
  type VoiceForgeModelStatus,
} from "@shared/speech-recommendation";
import {
  voiceForgeOrchestrator,
  type GenerateSpeechInput,
  type GenerateSpeechResult,
  type VoiceForgeJob,
} from "./voiceforge-orchestrator";
import type { DefaultVoiceMetadata } from "./voice-library-service";

export interface VoiceForgeMcpBackend {
  getModelStatuses(): VoiceForgeModelStatus[] | Promise<VoiceForgeModelStatus[]>;
  listVoices(): Promise<DefaultVoiceMetadata[]>;
  recommend(input: {
    characterCount: number;
    target?: "auto" | "local" | "agent";
    hasVoice?: boolean;
    speakerCount?: number;
    mode?: VoiceForgeMode;
    preference?: "speed" | "quality";
    needsInlinePauses?: boolean;
  }): SpeechRecommendation | Promise<SpeechRecommendation>;
  generate(input: GenerateSpeechInput): Promise<GenerateSpeechResult>;
  getJob(id: string): VoiceForgeJob | undefined | Promise<VoiceForgeJob | undefined>;
  cancelJob(id: string): VoiceForgeJob | undefined | Promise<VoiceForgeJob | undefined>;
  readJobAudio(id: string): Promise<Buffer | undefined>;
  readVoiceAudio?(id: string): Promise<{ audio: Buffer; mimeType: string; name: string } | undefined>;
}

const modelIdSchema = z.enum(VOICEFORGE_MODEL_IDS);
const modelChoiceSchema = z.enum(["auto", ...VOICEFORGE_MODEL_IDS]);
const targetSchema = z.enum(["local", "agent"]);
const targetPreferenceSchema = z.enum(["auto", "local", "agent"]);
const modeSchema = z.enum([
  "auto",
  "direct",
  "clone",
  "design",
  "preset",
  "continuation",
  "continuation-clone",
]);
const jobStateSchema = z.enum(["queued", "running", "completed", "failed", "cancelled"]);
const outputFormatSchema = z.enum(["wav", "mp3"]);
const outputMimeTypeSchema = z.enum(["audio/wav", "audio/mpeg"]);
const referenceEnhancementSchema = z.enum(["none", "cleanup", "audiosr"]);

const modelStatusOutputSchema = z.object({
  id: modelIdSchema,
  label: z.string(),
  repository_id: z.string(),
  targets: z.array(targetSchema),
  local_modes: z.array(modeSchema),
  agent_modes: z.array(modeSchema),
  local_character_limit: z.number().int(),
  agent_character_limit: z.number().int().optional(),
  local_ready: z.boolean(),
  agent_ready: z.boolean(),
  setup_hint: z.string().optional(),
});

const voiceOutputSchema = z.object({
  voice_id: z.string(),
  display_name: z.string(),
  format: z.string(),
  size_bytes: z.number().int(),
  has_transcript: z.boolean(),
  transcript: z.string().nullable(),
  preview_resource_uri: z.string(),
});

const candidateOutputSchema = z.object({
  model: modelIdSchema,
  target: targetSchema,
  mode: modeSchema.exclude(["auto"]),
  runnable_now: z.boolean(),
  blockers: z.array(z.string()),
});

const recommendationOutputSchema = z.object({
  character_count: z.number().int(),
  length_band: z.enum(["empty", "short", "medium", "long", "too-long"]),
  requested_target: targetPreferenceSchema,
  recommended: candidateOutputSchema.optional(),
  alternatives: z.array(candidateOutputSchema),
  reasons: z.array(z.string()),
  warnings: z.array(z.string()),
});

const jobOutputSchema = z.object({
  job_id: z.string(),
  model: modelIdSchema,
  target: targetSchema,
  mode: modeSchema.exclude(["auto"]),
  status: jobStateSchema,
  progress: z.number(),
  message: z.string().optional(),
  error: z.string().optional(),
  created_at: z.number(),
  updated_at: z.number(),
  output_format: outputFormatSchema,
  output_mime_type: outputMimeTypeSchema,
  chapter_count: z.number().int().nonnegative(),
  reference_enhancement: referenceEnhancementSchema,
  level_normalized: z.boolean(),
  audio_resource_uri: z.string().optional(),
  audio_url: z.string().optional(),
});

function textResult(structuredContent: Record<string, unknown>, summary?: string) {
  return {
    structuredContent,
    content: [{ type: "text" as const, text: summary || JSON.stringify(structuredContent) }],
  };
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/u, "");
}

function publicJob(job: VoiceForgeJob, publicBaseUrl: string) {
  return {
    job_id: job.id,
    model: job.model,
    target: job.target,
    mode: job.mode,
    status: job.status,
    progress: job.progress,
    message: job.message,
    error: job.error,
    created_at: job.createdAt,
    updated_at: job.updatedAt,
    output_format: job.outputFormat ?? "wav",
    output_mime_type: job.outputFormat === "mp3" ? "audio/mpeg" : "audio/wav",
    chapter_count: job.chapterCount ?? 0,
    reference_enhancement: job.referenceEnhancement ?? "none",
    level_normalized: job.levelNormalized ?? false,
    audio_resource_uri: job.audioResourceUri,
    audio_url: job.audioPath ? `${normalizeBaseUrl(publicBaseUrl)}${job.audioPath}` : undefined,
  };
}

function publicRecommendation(value: SpeechRecommendation) {
  const mapCandidate = (candidate: NonNullable<SpeechRecommendation["recommended"]>) => ({
    model: candidate.model,
    target: candidate.target,
    mode: candidate.mode,
    runnable_now: candidate.runnableNow,
    blockers: candidate.blockers,
  });
  return {
    character_count: value.characterCount,
    length_band: value.lengthBand,
    requested_target: value.requestedTarget,
    recommended: value.recommended ? mapCandidate(value.recommended) : undefined,
    alternatives: value.alternatives.map(mapCandidate),
    reasons: value.reasons,
    warnings: value.warnings,
  };
}

export function createVoiceForgeMcpServer(options?: {
  backend?: VoiceForgeMcpBackend;
  publicBaseUrl?: string;
}) {
  const backend = options?.backend ?? voiceForgeOrchestrator;
  const publicBaseUrl = options?.publicBaseUrl ?? `http://127.0.0.1:${process.env.PORT || "5000"}`;
  const server = new McpServer(
    { name: "voiceforge", version: "2.0.0" },
    {
      instructions:
        "Generate speech, not prose. If the user did not choose a model, call voiceforge_recommend_model first. Before generation, use an explicit target: Local keeps text/audio on this machine; Agent uploads the text and any reference voice to the official Hugging Face ZeroGPU Space. Use voiceforge_list_voices for safe voice IDs, then poll voiceforge_get_job until completion.",
    }
  );

  server.registerTool(
    "voiceforge_list_models",
    {
      title: "List VoiceForge speech models",
      description: "List every VoiceForge TTS model, supported Local/Agent modes, text limits, and current readiness.",
      outputSchema: { models: z.array(modelStatusOutputSchema) },
      annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
    },
    async () => {
      const statuses = await backend.getModelStatuses();
      const structuredContent = {
        models: statuses.map((status) => ({
          id: status.id,
          label: status.label,
          repository_id: status.repositoryId,
          targets: status.targets,
          local_modes: status.localModes,
          agent_modes: status.agentModes,
          local_character_limit: status.localCharacterLimit,
          agent_character_limit: status.agentCharacterLimit,
          local_ready: status.localReady,
          agent_ready: status.agentReady,
          setup_hint: status.setupHint,
        })),
      };
      return textResult(structuredContent, `VoiceForge exposes ${structuredContent.models.length} selectable speech models.`);
    }
  );

  server.registerTool(
    "voiceforge_list_voices",
    {
      title: "List default VoiceForge voices",
      description: "List safe IDs for local default voice recordings and their paired reference transcripts. No filesystem paths are returned.",
      outputSchema: { voices: z.array(voiceOutputSchema) },
      annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
    },
    async () => {
      const voices = await backend.listVoices();
      const structuredContent = {
        voices: voices.map((voice) => ({
          voice_id: voice.id,
          display_name: voice.name,
          format: voice.format,
          size_bytes: voice.sizeBytes,
          has_transcript: voice.hasTranscript,
          transcript: voice.transcript,
          preview_resource_uri: `voiceforge://default-voices/${voice.id}/audio`,
        })),
      };
      return textResult(structuredContent, `Found ${voices.length} default voices.`);
    }
  );

  server.registerTool(
    "voiceforge_recommend_model",
    {
      title: "Recommend a VoiceForge speech model",
      description: "Recommend a deterministic TTS model from text length, Local/Agent preference, voice availability, speaker count, mode, and speed/quality preference. This does not start inference.",
      inputSchema: z
        .object({
          text: z.string().max(500_000).optional().describe("Text that will be synthesized. Used only for its character count."),
          character_count: z.number().int().min(0).max(500_001).optional(),
          target: targetPreferenceSchema.default("auto"),
          has_voice: z.boolean().default(false),
          speaker_count: z.number().int().min(1).max(5).default(1),
          mode: modeSchema.default("auto"),
          preference: z.enum(["speed", "quality"]).default("quality"),
          needs_inline_pauses: z.boolean().default(false),
        })
        .refine((input) => input.text !== undefined || input.character_count !== undefined, {
          message: "Provide text or character_count.",
        }),
      outputSchema: recommendationOutputSchema,
      annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
    },
    async (input) => {
      const recommendation = await backend.recommend({
        characterCount: input.text?.length ?? input.character_count ?? 0,
        target: input.target,
        hasVoice: input.has_voice,
        speakerCount: input.speaker_count,
        mode: input.mode,
        preference: input.preference,
        needsInlinePauses: input.needs_inline_pauses,
      });
      const structuredContent = publicRecommendation(recommendation);
      return textResult(
        structuredContent,
        structuredContent.recommended
          ? `Recommended ${structuredContent.recommended.model} on ${structuredContent.recommended.target}.`
          : structuredContent.warnings[0] || "No compatible model was found."
      );
    }
  );

  server.registerTool(
    "voiceforge_generate_speech",
    {
      title: "Generate speech with VoiceForge",
      description: "Start an asynchronous TTS job. Local stays on this machine. Agent sends text and any selected default voice to the official Hugging Face Space and consumes ZeroGPU quota. Explicit model choices are honored or rejected, never silently replaced. Level normalization, MP3 export, exact chapters, and reference-audio enhancement are Qwen/MOSS-only; exact chapters additionally require Local, MP3 output, and [CHAPTER] markers.",
      inputSchema: {
        request_id: z.string().regex(/^[A-Za-z0-9_-]{8,128}$/).describe("Stable idempotency key; reuse it only when retrying the exact same request."),
        text: z.string().min(1).max(500_000),
        model: modelChoiceSchema.default("auto"),
        target: targetSchema.describe("Required privacy/compute choice: local or Hugging Face Agent."),
        mode: modeSchema.default("auto"),
        voice_id: z.string().optional(),
        voice_ids: z.array(z.string()).max(4).optional(),
        language: z.string().max(40).optional(),
        reference_text: z.string().max(20_000).optional(),
        voice_description: z.string().max(1_000).optional(),
        speaker: z.enum(["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]).optional(),
        instruction: z.string().max(1_000).optional(),
        preference: z.enum(["speed", "quality"]).default("quality"),
        style: z.string().max(500).optional(),
        guidance_scale: z.number().min(0.5).max(3).optional(),
        needs_inline_pauses: z.boolean().default(false),
        output_format: outputFormatSchema
          .optional()
          .describe("Qwen/MOSS only. Final managed audio format; defaults to wav."),
        use_chapters: z.boolean()
          .optional()
          .describe("Qwen/MOSS only. Exact MP3 chapters; requires Local, output_format=mp3, and [CHAPTER] markers. Defaults to false."),
        chapter_pause_ms: z.number().int().min(0).max(10_000)
          .optional()
          .describe("Qwen/MOSS only. Extra silence before each later chapter's spoken audio, in milliseconds; defaults to 0."),
        mp3_quality: z.number().int().min(0).max(9)
          .optional()
          .describe("Qwen/MOSS only. FFmpeg VBR quality for non-chaptered MP3 (0 best, 9 smallest); defaults to 2."),
        normalize_levels: z.boolean()
          .optional()
          .describe("Qwen/MOSS only. Normalize the assembled output to a consistent speech loudness target; defaults to true."),
        reference_enhancement: referenceEnhancementSchema
          .optional()
          .describe("Qwen/MOSS only. Enhance the selected reference voice with none, gentle FFmpeg cleanup, or isolated AudioSR; defaults to none."),
        audiosr_model: z.enum(["speech", "basic"])
          .optional()
          .describe("AudioSR model preset; defaults to speech."),
        audiosr_device: z.string().regex(/^(?:auto|cpu|mps|cuda(?::\d{1,3})?)$/u)
          .optional()
          .describe("AudioSR device: auto, cpu, mps, cuda, or cuda:N (up to three digits); defaults to auto."),
        audiosr_ddim_steps: z.number().int().min(10).max(250)
          .optional()
          .describe("AudioSR DDIM steps; defaults to 50."),
        audiosr_guidance_scale: z.number().min(1).max(10)
          .optional()
          .describe("AudioSR classifier-free guidance scale; defaults to 3.5."),
        audiosr_seed: z.number().int().min(-2_147_483_648).max(2_147_483_647)
          .optional()
          .describe("AudioSR random seed; defaults to 42."),
      },
      outputSchema: {
        job: jobOutputSchema,
        selected_by_recommendation: z.boolean(),
        recommendation_reasons: z.array(z.string()),
        warnings: z.array(z.string()),
      },
      annotations: { readOnlyHint: false, openWorldHint: true, destructiveHint: false },
    },
    async (input) => {
      if (input.voice_id && input.voice_ids?.length) {
        throw new Error("Use voice_id or voice_ids, not both.");
      }
      const result = await backend.generate({
        requestId: input.request_id,
        text: input.text,
        model: input.model,
        target: input.target,
        mode: input.mode,
        voiceId: input.voice_id,
        voiceIds: input.voice_ids,
        language: input.language,
        referenceText: input.reference_text,
        voiceDescription: input.voice_description,
        speaker: input.speaker,
        instruction: input.instruction,
        preference: input.preference,
        style: input.style,
        guidanceScale: input.guidance_scale,
        needsInlinePauses: input.needs_inline_pauses,
        outputFormat: input.output_format,
        useChapters: input.use_chapters,
        chapterPauseMs: input.chapter_pause_ms,
        mp3Quality: input.mp3_quality,
        normalizeLevels: input.normalize_levels,
        referenceEnhancement: input.reference_enhancement,
        audioSrModel: input.audiosr_model,
        audioSrDevice: input.audiosr_device,
        audioSrDdimSteps: input.audiosr_ddim_steps,
        audioSrGuidanceScale: input.audiosr_guidance_scale,
        audioSrSeed: input.audiosr_seed,
      });
      const job = publicJob(result.job, publicBaseUrl);
      const structuredContent = {
        job,
        selected_by_recommendation: result.selectedByRecommendation,
        recommendation_reasons: result.recommendationReasons,
        warnings: result.warnings,
      };
      return textResult(structuredContent, `Started ${job.model} job ${job.job_id} on ${job.target}.`);
    }
  );

  server.registerTool(
    "voiceforge_get_job",
    {
      title: "Get a VoiceForge speech job",
      description: "Read current synthesis progress. Poll this tool until the job is completed, failed, or cancelled.",
      inputSchema: { job_id: z.string() },
      outputSchema: { job: jobOutputSchema },
      annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
    },
    async ({ job_id }) => {
      const value = await backend.getJob(job_id);
      if (!value) throw new Error("VoiceForge job not found.");
      const job = publicJob(value, publicBaseUrl);
      const result = textResult({ job }, `Job ${job.job_id} is ${job.status} (${Math.round(job.progress)}%).`);
      if (job.status === "completed" && job.audio_resource_uri) {
        result.content.push({
          type: "resource_link" as const,
          uri: job.audio_resource_uri,
          name: `${job.model}-${job.job_id}.${job.output_format}`,
          mimeType: job.output_mime_type,
          description: "Generated VoiceForge speech audio",
        } as never);
      }
      return result;
    }
  );

  server.registerTool(
    "voiceforge_cancel_job",
    {
      title: "Cancel a VoiceForge speech job",
      description: "Cancel a queued or running VoiceForge TTS job. Completed or failed jobs are left unchanged.",
      inputSchema: { job_id: z.string() },
      outputSchema: { job: jobOutputSchema },
      annotations: { readOnlyHint: false, openWorldHint: false, destructiveHint: true },
    },
    async ({ job_id }) => {
      const value = await backend.cancelJob(job_id);
      if (!value) throw new Error("VoiceForge job not found.");
      const job = publicJob(value, publicBaseUrl);
      return textResult({ job }, `Job ${job.job_id} is ${job.status}.`);
    }
  );

  server.registerResource(
    "generated-speech-audio",
    new ResourceTemplate("voiceforge://speech/jobs/{jobId}/audio", { list: undefined }),
    { description: "Completed VoiceForge speech audio" },
    async (uri, variables) => {
      const jobId = String(variables.jobId || "");
      const job = await backend.getJob(jobId);
      if (!job || job.status !== "completed") {
        throw new Error("Generated audio is not ready or no longer available.");
      }
      const audio = await backend.readJobAudio(jobId);
      if (!audio) throw new Error("Generated audio is not ready or no longer available.");
      return {
        contents: [{
          uri: uri.href,
          mimeType: job.outputFormat === "mp3" ? "audio/mpeg" : "audio/wav",
          blob: audio.toString("base64"),
        }],
      };
    }
  );

  server.registerResource(
    "default-voice-audio",
    new ResourceTemplate("voiceforge://default-voices/{voiceId}/audio", { list: undefined }),
    { description: "VoiceForge default reference voice preview" },
    async (uri, variables) => {
      const voiceId = String(variables.voiceId || "");
      const voice = backend.readVoiceAudio ? await backend.readVoiceAudio(voiceId) : undefined;
      if (!voice) throw new Error("Default voice was not found or is no longer safe to read.");
      return { contents: [{ uri: uri.href, mimeType: voice.mimeType, blob: voice.audio.toString("base64") }] };
    }
  );

  return server;
}
