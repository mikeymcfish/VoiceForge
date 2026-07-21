import { z } from "zod";
import { MAX_BATCH_SIZE } from "./batch-profiles";

// Processing Job Schema
export const processingJobSchema = z.object({
  id: z.string(),
  fileName: z.string(),
  fileType: z.enum(["txt", "epub"]),
  originalText: z.string(),
  processedText: z.string(),
  status: z.enum(["idle", "processing", "completed", "failed", "cancelled"]),
  progress: z.number().min(0).max(100),
  currentChunk: z.number(),
  totalChunks: z.number(),
  createdAt: z.date(),
  completedAt: z.date().optional(),
});

export type ProcessingJob = z.infer<typeof processingJobSchema>;

// Text Cleaning Options
export const cleaningOptionsSchema = z.object({
  replaceSmartQuotes: z.boolean().default(true),
  fixOcrErrors: z.boolean().default(true),
  correctSpelling: z.boolean().default(false),
  removeUrls: z.boolean().default(true),
  removeFootnotes: z.boolean().default(true),
  addPunctuation: z.boolean().default(true),
  insertChapterBreaks: z.boolean().default(false),
  fixHyphenation: z.boolean().default(false),
});

export type CleaningOptions = z.infer<typeof cleaningOptionsSchema>;

// Multi-Speaker Configuration
export const speakerConfigSchema = z.object({
  mode: z.enum(["none", "format", "intelligent"]),
  speakerCount: z.number().min(1).max(20).default(2),
  labelFormat: z.enum(["speaker", "bracket"]),
  speakerMapping: z.record(z.string(), z.string()).optional(),
  extractCharacters: z.boolean().default(false),
  sampleSize: z.number().min(5).max(160).default(50),
  includeNarrator: z.boolean().default(false),
  narratorAttribution: z
    .enum(["remove", "verbatim", "contextual"])
    .default("remove"),
  characterMapping: z
    .array(
      z.object({
        name: z.string(),
        speakerNumber: z.number(),
      })
    )
    .optional(),
  narratorCharacterName: z.string().optional(),
});

export type SpeakerConfig = z.infer<typeof speakerConfigSchema>;

// Model Source
export const modelSourceSchema = z.enum(["api", "ollama"]);
export type ModelSource = z.infer<typeof modelSourceSchema>;

// Processing Configuration
export const processingConfigSchema = z.object({
  batchSize: z.number().int().min(1).max(MAX_BATCH_SIZE).default(10),
  cleaningOptions: cleaningOptionsSchema,
  speakerConfig: speakerConfigSchema.optional(),
  modelSource: modelSourceSchema.default("api"),
  modelName: z.string().default("meta-llama/Meta-Llama-3.1-8B-Instruct"),
  localModelName: z.string().optional(),
  ollamaModelName: z.string().optional(),
  ollamaThinkingEnabled: z.boolean().default(false),
  ollamaContextWindow: z.number().int().min(2_048).max(65_536).default(8_192).optional(),
  ollamaMaxOutputTokens: z.number().int().min(256).max(16_384).default(2_000).optional(),
  temperature: z.number().min(0).max(2).default(0.3).optional(),
  llmCleaningDisabled: z.boolean().default(false).optional(),
  customInstructions: z.string().optional(),
  singlePass: z.boolean().default(false),
  extendedExamples: z.boolean().default(false),
});

export type ProcessingConfig = z.infer<typeof processingConfigSchema>;

// Activity Log Entry
export const logEntrySchema = z.object({
  id: z.string(),
  timestamp: z.date(),
  type: z.enum(["info", "success", "warning", "error"]),
  message: z.string(),
  details: z.string().optional(),
});

export type LogEntry = z.infer<typeof logEntrySchema>;

// File Upload Response
export const fileUploadResponseSchema = z.object({
  fileName: z.string(),
  fileType: z.string(),
  text: z.string(),
  wordCount: z.number(),
  charCount: z.number(),
});

export type FileUploadResponse = z.infer<typeof fileUploadResponseSchema>;

// Processing Request
export const processTextRequestSchema = z.object({
  text: z.string().min(1).max(5_000_000),
  config: processingConfigSchema,
});

export type ProcessTextRequest = z.infer<typeof processTextRequestSchema>;

// Deterministic text cleaning (no LLM)
export const deterministicCleanRequestSchema = z.object({
  text: z.string(),
  options: cleaningOptionsSchema,
});

export type DeterministicCleanRequest = z.infer<typeof deterministicCleanRequestSchema>;

export const deterministicCleanResponseSchema = z.object({
  cleanedText: z.string(),
  appliedSteps: z.array(z.string()).default([]),
});

export type DeterministicCleanResponse = z.infer<typeof deterministicCleanResponseSchema>;

// Processing Response (chunk result)
export const processChunkResponseSchema = z.object({
  chunkIndex: z.number(),
  processedText: z.string(),
  status: z.enum(["success", "retry", "failed"]),
  retryCount: z.number().default(0),
});

export type ProcessChunkResponse = z.infer<typeof processChunkResponseSchema>;

// WebSocket Message Types
export const wsMessageSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("progress"),
    payload: z.object({
      progress: z.number(),
      currentChunk: z.number(),
      totalChunks: z.number(),
      lastChunkMs: z.number().optional(),
      avgChunkMs: z.number().optional(),
      etaMs: z.number().optional(),
      inputTokens: z.number().optional(),
      outputTokens: z.number().optional(),
      inputCost: z.number().optional(),
      outputCost: z.number().optional(),
      totalInputTokens: z.number().optional(),
      totalOutputTokens: z.number().optional(),
      totalCost: z.number().optional(),
    }),
  }),
  z.object({
    type: z.literal("chunk"),
    payload: processChunkResponseSchema,
  }),
  z.object({
    type: z.literal("log"),
    payload: logEntrySchema,
  }),
  z.object({
    type: z.literal("complete"),
    payload: z.object({
      processedText: z.string(),
      totalChunks: z.number(),
      failedChunks: z.number().optional(),
      failedChunkIndexes: z.array(z.number()).optional(),
      totalInputTokens: z.number().optional(),
      totalOutputTokens: z.number().optional(),
      totalCost: z.number().optional(),
    }),
  }),
  z.object({
    type: z.literal("error"),
    payload: z.object({
      message: z.string(),
      details: z.string().optional(),
    }),
  }),
]);

export type WSMessage = z.infer<typeof wsMessageSchema>;

// PDF OCR integration schemas
export const pdfOcrJobStatusSchema = z.object({
  id: z.string(),
  status: z.enum(["queued", "running", "completed", "failed", "cancelled"]),
  progress: z.number().min(0).max(100),
  message: z.string().optional(),
  pageCount: z.number().optional(),
  processedPages: z.number().optional(),
  outputFile: z.string().optional(),
  pdfFileName: z.string().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
  error: z.string().optional(),
});

export type PdfOcrJobStatus = z.infer<typeof pdfOcrJobStatusSchema>;

export const pdfOcrLogEntrySchema = z.object({
  id: z.string(),
  jobId: z.string(),
  level: z.enum(["info", "warn", "error"]),
  message: z.string(),
  timestamp: z.number(),
});

export type PdfOcrLogEntry = z.infer<typeof pdfOcrLogEntrySchema>;

export const pdfOcrDownloadStatusSchema = z.enum(["idle", "in-progress", "completed", "failed"]);
export type PdfOcrDownloadStatus = z.infer<typeof pdfOcrDownloadStatusSchema>;

export const pdfOcrConfigSchema = z.object({
  pythonPath: z.string().optional(),
  deepseekRepoPath: z.string().optional(),
  huggingFaceRepoId: z.string().optional(),
  huggingFaceRevision: z.string().optional(),
  lastResolvedModulePath: z.string().optional(),
});

export type PdfOcrConfig = z.infer<typeof pdfOcrConfigSchema>;

export const pdfOcrStatusSchema = z.object({
  jobs: z.array(pdfOcrJobStatusSchema),
  modelsDir: z.string(),
  downloadStatus: pdfOcrDownloadStatusSchema,
  downloadError: z.string().optional(),
  config: pdfOcrConfigSchema,
});

export type PdfOcrStatus = z.infer<typeof pdfOcrStatusSchema>;

export const pdfOcrWsMessageSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("status"),
    payload: pdfOcrStatusSchema,
  }),
  z.object({
    type: z.literal("job"),
    payload: pdfOcrJobStatusSchema,
  }),
  z.object({
    type: z.literal("log"),
    payload: pdfOcrLogEntrySchema,
  }),
  z.object({
    type: z.literal("text"),
    payload: z.object({
      jobId: z.string(),
      text: z.string(),
    }),
  }),
]);

export type PdfOcrWsMessage = z.infer<typeof pdfOcrWsMessageSchema>;

// IndexTTS integration schemas
export const ttsDownloadStatusSchema = z.enum(["idle", "in-progress", "completed", "failed"]);
export type TtsDownloadStatus = z.infer<typeof ttsDownloadStatusSchema>;

export const ttsJobStatusSchema = z.object({
  id: z.string(),
  status: z.enum(["queued", "running", "completed", "failed", "cancelled"]),
  progress: z.number().min(0).max(100),
  message: z.string().optional(),
  outputFile: z.string().optional(),
  voiceFileName: z.string().optional(),
  textFileName: z.string().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
  error: z.string().optional(),
});

export type TtsJobStatus = z.infer<typeof ttsJobStatusSchema>;

export const ttsStatusSchema = z.object({
  downloadStatus: ttsDownloadStatusSchema,
  loadStatus: ttsDownloadStatusSchema,
  modelsReady: z.boolean(),
  runtimeConfigured: z.boolean(),
  modelsPath: z.string(),
  lastDownloadError: z.string().optional(),
  lastLoadError: z.string().optional(),
  jobs: z.array(ttsJobStatusSchema),
});

export type TtsStatus = z.infer<typeof ttsStatusSchema>;

export const ttsWsMessageSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("status"),
    payload: ttsStatusSchema,
  }),
  z.object({
    type: z.literal("job"),
    payload: ttsJobStatusSchema,
  }),
  z.object({
    type: z.literal("log"),
    payload: z.object({
      id: z.string(),
      level: z.enum(["info", "warn", "error"]),
      message: z.string(),
      timestamp: z.number(),
    }),
  }),
]);

export type TtsWsMessage = z.infer<typeof ttsWsMessageSchema>;

// VibeVoice integration schemas
export const vibevoiceSetupStatusSchema = z.enum([
  "idle",
  "in-progress",
  "completed",
  "failed",
]);

export type VibevoiceSetupStatus = z.infer<typeof vibevoiceSetupStatusSchema>;

export const vibevoiceJobStatusSchema = z.object({
  id: z.string(),
  status: z.enum(["queued", "running", "completed", "failed", "cancelled"]),
  progress: z.number().min(0).max(100),
  message: z.string().optional(),
  outputFile: z.string().optional(),
  voiceFileName: z.string().optional(),
  voiceFileNames: z.array(z.string()).optional(),
  textFileName: z.string().optional(),
  style: z.string().optional(),
  selectedModel: z.string().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
  error: z.string().optional(),
});

export type VibevoiceJobStatus = z.infer<typeof vibevoiceJobStatusSchema>;

export const vibevoiceStatusSchema = z.object({
  setupStatus: vibevoiceSetupStatusSchema,
  ready: z.boolean(),
  repoPath: z.string(),
  lastSetupError: z.string().optional(),
  availableModels: z.array(z.object({ id: z.string(), path: z.string() })),
  jobs: z.array(vibevoiceJobStatusSchema),
});

export type VibevoiceStatus = z.infer<typeof vibevoiceStatusSchema>;

export const vibevoiceWsMessageSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("status"),
    payload: vibevoiceStatusSchema,
  }),
  z.object({
    type: z.literal("job"),
    payload: vibevoiceJobStatusSchema,
  }),
  z.object({
    type: z.literal("log"),
    payload: z.object({
      id: z.string(),
      level: z.enum(["info", "warn", "error"]),
      message: z.string(),
      timestamp: z.number(),
    }),
  }),
]);

export type VibevoiceWsMessage = z.infer<typeof vibevoiceWsMessageSchema>;

// HuggingFace token management
export const huggingFaceTokenStatusSchema = z.object({
  configured: z.boolean(),
  tokenPreview: z.string().optional(),
});

export type HuggingFaceTokenStatus = z.infer<typeof huggingFaceTokenStatusSchema>;

const huggingFaceTokenValueSchema = z
  .string()
  .max(512)
  .refine((value) => !/[\u0000-\u001f\u007f]/.test(value), {
    message: "Hugging Face tokens cannot contain control characters",
  })
  .transform((value) => value.trim())
  .refine((value) => value.length === 0 || /^hf_[A-Za-z0-9_-]{20,256}$/.test(value), {
    message: "Enter a valid Hugging Face access token",
  });

export const huggingFaceTokenUpdateSchema = z.object({
  token: huggingFaceTokenValueSchema.optional().nullable(),
});

export type HuggingFaceTokenUpdate = z.infer<typeof huggingFaceTokenUpdateSchema>;

// Additional speech engines (Qwen3-TTS and MOSS-TTS) share one job/status
// contract. IndexTTS and VibeVoice keep their existing compatibility APIs.
export const speechEngineSchema = z.enum(["qwen", "moss"]);
export type SpeechEngine = z.infer<typeof speechEngineSchema>;

export const speechExecutionTargetSchema = z.enum(["local", "hf-space"]);
export type SpeechExecutionTarget = z.infer<typeof speechExecutionTargetSchema>;

export const speechOutputFormatSchema = z.enum(["wav", "mp3"]);
export type SpeechOutputFormat = z.infer<typeof speechOutputFormatSchema>;

export const speechReferenceEnhancementSchema = z.enum(["none", "cleanup", "audiosr"]);
export type SpeechReferenceEnhancement = z.infer<typeof speechReferenceEnhancementSchema>;

export const speechReviewSegmentSchema = z.object({
  index: z.number().int().nonnegative(),
  text: z.string(),
  durationSeconds: z.number().positive(),
  attempt: z.number().int().positive(),
  startsChapter: z.boolean().optional(),
  chapterTitle: z.string().optional(),
  paceRatio: z.number().positive().optional(),
  paceStatus: z
    .enum(["typical", "unusually-fast", "unusually-slow", "not-compared"])
    .optional(),
  updatedAt: z.number().int().nonnegative(),
});

export type SpeechReviewSegment = z.infer<typeof speechReviewSegmentSchema>;

export const speechJobStatusSchema = z.object({
  id: z.string(),
  engine: speechEngineSchema,
  target: speechExecutionTargetSchema,
  mode: z.string(),
  status: z.enum([
    "queued",
    "running",
    "awaiting-review",
    "completed",
    "failed",
    "cancelled",
  ]),
  progress: z.number().min(0).max(100),
  message: z.string().optional(),
  outputFile: z.string().optional(),
  voiceFileName: z.string().optional(),
  modelId: z.string().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
  error: z.string().optional(),
  queuePosition: z.number().int().nonnegative().optional(),
  etaSeconds: z.number().nonnegative().optional(),
  outputFormat: speechOutputFormatSchema.optional(),
  outputMimeType: z.enum(["audio/wav", "audio/mpeg"]).optional(),
  chapterCount: z.number().int().nonnegative().optional(),
  referenceEnhancement: speechReferenceEnhancementSchema.optional(),
  levelNormalized: z.boolean().optional(),
  reviewSegmentCount: z.number().int().positive().optional(),
  reviewRevision: z.number().int().nonnegative().optional(),
  reviewError: z.string().optional(),
});

export type SpeechJobStatus = z.infer<typeof speechJobStatusSchema>;

export const speechEngineRuntimeStatusSchema = z.object({
  engine: speechEngineSchema,
  setupStatus: z.enum(["idle", "in-progress", "completed", "failed"]),
  setupProgress: z.number().min(0).max(100).optional(),
  setupMessage: z.string().optional(),
  setupUpdatedAt: z.number().int().nonnegative().optional(),
  setupStalled: z.boolean().optional(),
  runtimeConfigured: z.boolean(),
  modelsReady: z.boolean(),
  availableModels: z.array(z.string()),
  modelsPath: z.string(),
  lastSetupError: z.string().optional(),
  spaceId: z.string(),
  hostedAvailable: z.boolean(),
  localModes: z.array(z.string()),
  hostedModes: z.array(z.string()),
});

export type SpeechEngineRuntimeStatus = z.infer<typeof speechEngineRuntimeStatusSchema>;

export const speechStatusSchema = z.object({
  tokenConfigured: z.boolean(),
  audioProcessing: z.object({
    ffmpegAvailable: z.boolean(),
    audioSrAvailable: z.boolean(),
  }),
  engines: z.array(speechEngineRuntimeStatusSchema),
  jobs: z.array(speechJobStatusSchema),
});

export type SpeechStatus = z.infer<typeof speechStatusSchema>;

export const speechWsMessageSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("status"), payload: speechStatusSchema }),
  z.object({ type: z.literal("job"), payload: speechJobStatusSchema }),
  z.object({
    type: z.literal("log"),
    payload: z.object({
      id: z.string(),
      level: z.enum(["info", "warn", "error"]),
      message: z.string(),
      timestamp: z.number(),
    }),
  }),
]);

export type SpeechWsMessage = z.infer<typeof speechWsMessageSchema>;

export const huggingFaceUsageMetricSchema = z.object({
  status: z.enum(["reported", "estimated", "unavailable"]),
  authoritative: z.boolean(),
  unit: z.enum(["seconds", "usd"]),
  limit: z.number().nonnegative().optional(),
  used: z.number().nonnegative().optional(),
  remaining: z.number().nonnegative().optional(),
  resetAt: z.number().optional(),
  message: z.string(),
});

export type HuggingFaceUsageMetric = z.infer<typeof huggingFaceUsageMetricSchema>;

export const huggingFaceUsageStatusSchema = z.object({
  tokenConfigured: z.boolean(),
  accountName: z.string().optional(),
  plan: z.string().optional(),
  fetchedAt: z.number(),
  zeroGpu: huggingFaceUsageMetricSchema,
  inferenceCredits: huggingFaceUsageMetricSchema,
});

export type HuggingFaceUsageStatus = z.infer<typeof huggingFaceUsageStatusSchema>;

export const defaultVoiceSchema = z.object({
  id: z.string(),
  displayName: z.string(),
  format: z.string(),
  sizeBytes: z.number().int().positive(),
  hasTranscript: z.boolean(),
  transcript: z.string().optional(),
});

export type DefaultVoice = z.infer<typeof defaultVoiceSchema>;

export const defaultVoiceCatalogSchema = z.object({
  voices: z.array(defaultVoiceSchema),
  warnings: z.array(z.string()).default([]),
});

export type DefaultVoiceCatalog = z.infer<typeof defaultVoiceCatalogSchema>;
