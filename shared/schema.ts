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
  // Merge words split by line breaks/hyphens (PDF/EPUB artifacts)
  fixHyphenation: z.boolean().default(false),
});

export type CleaningOptions = z.infer<typeof cleaningOptionsSchema>;

// Multi-Speaker Configuration
export const speakerConfigSchema = z.object({
  mode: z.enum(["none", "format", "intelligent"]), // "none" = single speaker, no tags
  speakerCount: z.number().min(1).max(20).default(2),
  labelFormat: z.enum(["speaker", "bracket"]), // "Speaker 1:" or "[1]:"
  speakerMapping: z.record(z.string(), z.string()).optional(), // detected name -> speaker label
  extractCharacters: z.boolean().default(false), // Whether to extract character names
  sampleSize: z.number().min(5).max(100).default(20), // Number of sentences for character extraction
  includeNarrator: z.boolean().default(false), // Include narrator as separate speaker
  // How to handle dialogue attribution tags like "he said", "she replied" when Narrator is included
  narratorAttribution: z
    .enum(["remove", "verbatim", "contextual"]) // remove = strip tags; verbatim = narrator reads tag as-is; contextual = narrator summarizes action/context
    .default("remove"),
  characterMapping: z.array(z.object({ // Extracted character name to speaker number mapping
    name: z.string(),
    speakerNumber: z.number(),
  })).optional(),
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
  modelSource: modelSourceSchema.default("api"), // API or Local
  // Default to a model available on Hugging Face Inference provider
  modelName: z.string().default("meta-llama/Meta-Llama-3.1-8B-Instruct"), // For API
  localModelName: z.string().optional(), // For local models
  ollamaModelName: z.string().optional(), // For Ollama models
  customInstructions: z.string().optional(), // Custom instructions for the LLM
  // If true, run cleaning + speaker formatting in one LLM call per chunk
  singlePass: z.boolean().default(false),
  // Use shorter instruction prompts to reduce tokens
  concisePrompts: z.boolean().default(true),
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
      lastChunkMs: z.number().optional(),
      avgChunkMs: z.number().optional(),
      etaMs: z.number().optional(),
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
