import type { ProcessingConfig } from "@shared/schema";
import { llmService } from "./llm-service";

export interface ProcessingProgress {
  chunkIndex: number;
  processedText: string;
  status: "success" | "retry" | "failed";
  retryCount: number;
  lastChunkMs?: number;
  avgChunkMs?: number;
  etaMs?: number;
}

export class TextProcessor {
  private splitIntoChunks(text: string, batchSize: number): string[] {
    // Split by sentences (basic sentence detection)
    // Include sentences that may not end with punctuation and preserve leading fragments
    const sentences = text.match(/[^.!?]+(?:[.!?]+|$)/g) || [text];
    
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    
    for (const sentence of sentences) {
      currentChunk.push(sentence.trim());
      
      if (currentChunk.length >= batchSize) {
        chunks.push(currentChunk.join(" "));
        currentChunk = [];
      }
    }
    
    // Add remaining sentences
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join(" "));
    }
    
    return chunks.filter((chunk) => chunk.length > 0);
  }

  async processText(
    text: string,
    config: ProcessingConfig,
    onProgress: (progress: ProcessingProgress) => void
  ): Promise<string> {
    const chunks = this.splitIntoChunks(text, config.batchSize);
    const processedChunks: string[] = [];
    const totalChunks = chunks.length;
    let durationsTotal = 0;
    let processedCount = 0;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      let retryCount = 0;
      let success = false;
      let processedText = "";
      const chunkStart = Date.now();

      // Try processing with one retry on failure
      while (retryCount < 2 && !success) {
        try {
          processedText = await llmService.processChunk({
            text: chunk,
            cleaningOptions: config.cleaningOptions,
            speakerConfig: config.speakerConfig,
            modelSource: config.modelSource,
            modelName: config.modelName,
            ollamaModelName: (config as any).ollamaModelName,
            customInstructions: config.customInstructions,
            singlePass: (config as any).singlePass === true,
            concisePrompts: (config as any).concisePrompts !== false,
          });

          // Validate output
          const validation = await llmService.validateOutput(
            chunk,
            processedText,
            config.modelSource
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
              await new Promise((resolve) => setTimeout(resolve, 1000));
            }
          }
        } catch (error) {
          retryCount++;
          if (retryCount < 2) {
            onProgress({
              chunkIndex: i,
              processedText: "",
              status: "retry",
              retryCount,
            });
            
            await new Promise((resolve) => setTimeout(resolve, 1000));
          }
        }
      }

      // If still failed after retries, use original chunk
      if (!success) {
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

    return processedChunks.join("\n\n");
  }
}

export const textProcessor = new TextProcessor();
