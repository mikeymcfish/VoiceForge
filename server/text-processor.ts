import type { ProcessingConfig } from "@shared/schema";
import { chunkTextPreservingStructure } from "@shared/text-utils";
import { llmService } from "./llm-service";

export interface ProcessingProgress {
  chunkIndex: number;
  processedText: string;
  status: "success" | "retry" | "failed";
  retryCount: number;
  lastChunkMs?: number;
  avgChunkMs?: number;
  etaMs?: number;
  // Token/cost metrics for this chunk (optional)
  inputTokens?: number;
  outputTokens?: number;
  inputCost?: number;
  outputCost?: number;
  totalInputTokens?: number;
  totalOutputTokens?: number;
  totalCost?: number;
}

export interface TextProcessingResult {
  text: string;
  totalChunks: number;
  failedChunkIndexes: number[];
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCost: number;
}

export class ProcessingCancelledError extends Error {
  constructor() {
    super("Processing cancelled");
    this.name = "ProcessingCancelledError";
  }
}

function waitForRetry(signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new ProcessingCancelledError());
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new ProcessingCancelledError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, 1000);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export class TextProcessor {
  async processText(
    text: string,
    config: ProcessingConfig,
    onProgress: (progress: ProcessingProgress) => void,
    shouldCancel: () => boolean = () => false,
    signal?: AbortSignal
  ): Promise<TextProcessingResult> {
    const chunks = chunkTextPreservingStructure(text, config.batchSize);
    const processedChunks: string[] = [];
    const totalChunks = chunks.length;
    let durationsTotal = 0;
    let processedCount = 0;
    let totalInTokens = 0, totalOutTokens = 0, totalCost = 0;
    const failedChunkIndexes: number[] = [];

    for (let i = 0; i < chunks.length; i++) {
      if (shouldCancel()) throw new ProcessingCancelledError();
      const chunk = chunks[i].text;
      let retryCount = 0;
      let success = false;
      let processedText = "";
      const chunkStart = Date.now();

      // Try processing with one retry on failure
      while (retryCount < 2 && !success) {
        if (shouldCancel()) throw new ProcessingCancelledError();
        try {
          const result = await llmService.processChunk({
            text: chunk,
            cleaningOptions: config.cleaningOptions,
            speakerConfig: config.speakerConfig,
            modelSource: config.modelSource,
            modelName: config.modelName,
            ollamaModelName: (config as any).ollamaModelName,
            temperature: (config as any).temperature,
            llmCleaningDisabled: (config as any).llmCleaningDisabled === true,
            customInstructions: config.customInstructions,
            singlePass: (config as any).singlePass === true,
            extendedExamples: (config as any).extendedExamples === true,
            signal,
          });
          processedText = result.text;
          const chunkIn = result.usage.inputTokens || 0;
          const chunkOut = result.usage.outputTokens || 0;
          const chunkCost = (result.usage.inputCost || 0) + (result.usage.outputCost || 0);
          totalInTokens += chunkIn; totalOutTokens += chunkOut; totalCost += chunkCost;

          // Validate output
          const validation = await llmService.validateOutput(
            chunk,
            processedText,
            config
          );

          if (validation.valid) {
            success = true;
            processedChunks.push(processedText);
            const lastChunkMs = Date.now() - chunkStart;
            durationsTotal += lastChunkMs;
            processedCount += 1;
            const avgChunkMs = durationsTotal / processedCount;
            const remaining = totalChunks - (i + 1);
            const etaMs = Math.max(0, Math.round(avgChunkMs * remaining));
            
            onProgress({
              chunkIndex: i,
              processedText,
              status: "success",
              retryCount,
              lastChunkMs,
              avgChunkMs,
              etaMs,
              inputTokens: chunkIn,
              outputTokens: chunkOut,
              inputCost: result.usage.inputCost,
              outputCost: result.usage.outputCost,
              totalInputTokens: totalInTokens,
              totalOutputTokens: totalOutTokens,
              totalCost,
            });
          } else {
            retryCount++;
            if (retryCount < 2) {
              onProgress({
                chunkIndex: i,
                processedText: "",
                status: "retry",
                retryCount,
              });
              
              // Wait a bit before retry
              await waitForRetry(signal);
              if (shouldCancel()) throw new ProcessingCancelledError();
            }
          }
        } catch (error) {
          if (
            error instanceof ProcessingCancelledError ||
            signal?.aborted ||
            (error instanceof Error && error.name === "AbortError")
          ) {
            throw new ProcessingCancelledError();
          }
          retryCount++;
          if (retryCount < 2) {
            onProgress({
              chunkIndex: i,
              processedText: "",
              status: "retry",
              retryCount,
            });
            
            await waitForRetry(signal);
            if (shouldCancel()) throw new ProcessingCancelledError();
          }
        }
      }

      // If still failed after retries, use original chunk
      if (!success) {
        failedChunkIndexes.push(i);
        processedChunks.push(chunk);
        const lastChunkMs = Date.now() - chunkStart;
        durationsTotal += lastChunkMs;
        processedCount += 1;
        const avgChunkMs = durationsTotal / processedCount;
        const remaining = totalChunks - (i + 1);
        const etaMs = Math.max(0, Math.round(avgChunkMs * remaining));
        onProgress({
          chunkIndex: i,
          processedText: chunk,
          status: "failed",
          retryCount,
          lastChunkMs,
          avgChunkMs,
          etaMs,
        });
      }
    }

    return {
      text: processedChunks
        .map((processed, index) => `${processed}${chunks[index]?.separatorAfter ?? ""}`)
        .join(""),
      totalChunks,
      failedChunkIndexes,
      totalInputTokens: totalInTokens,
      totalOutputTokens: totalOutTokens,
      totalCost,
    };
  }
}

export const textProcessor = new TextProcessor();
