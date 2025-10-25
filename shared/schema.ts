import { z } from "zod";

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
  batchSize: z.number().min(1).max(50).default(10),
  cleaningOptions: cleaningOptionsSchema,
  speakerConfig: speakerConfigSchema.optional(),
  modelSource: modelSourceSchema.default("api"),
  modelName: z.string().default("meta-llama/Meta-Llama-3.1-8B-Instruct"),
  localModelName: z.string().optional(),
  ollamaModelName: z.string().optional(),
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
  text: z.string(),
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
  status: z.enum(["queued", "running", "completed", "failed"]),
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

export const pdfOcrStatusSchema = z.object({
  jobs: z.array(pdfOcrJobStatusSchema),
  modelsDir: z.string(),
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
  status: z.enum(["queued", "running", "completed", "failed"]),
  progress: z.number().min(0).max(100),
  message: z.string().optional(),
  steps: z.number().optional(),
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
  status: z.enum(["queued", "running", "completed", "failed"]),
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

export const huggingFaceTokenUpdateSchema = z.object({
  token: z.string().optional().nullable(),
});

export type HuggingFaceTokenUpdate = z.infer<typeof huggingFaceTokenUpdateSchema>;
