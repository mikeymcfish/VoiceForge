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
});

export type CleaningOptions = z.infer<typeof cleaningOptionsSchema>;

// Multi-Speaker Configuration
export const speakerConfigSchema = z.object({
  mode: z.enum(["format", "intelligent"]),
  speakerCount: z.number().min(1).max(20).default(2),
  labelFormat: z.enum(["speaker", "bracket"]), // "Speaker 1:" or "[1]:"
  speakerMapping: z.record(z.string(), z.string()).optional(), // detected name -> speaker label
});

export type SpeakerConfig = z.infer<typeof speakerConfigSchema>;

// Processing Configuration
export const processingConfigSchema = z.object({
  batchSize: z.number().min(1).max(50).default(10),
  cleaningOptions: cleaningOptionsSchema,
  speakerConfig: speakerConfigSchema.optional(),
  modelName: z.string().default("Qwen/Qwen2.5-72B-Instruct"),
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
