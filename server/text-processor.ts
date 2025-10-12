import type { ProcessingConfig } from "@shared/schema";
import { llmService } from "./llm-service";

export interface ProcessingProgress {
  chunkIndex: number;
  processedText: string;
  status: "success" | "retry" | "failed";
  retryCount: number;
}

export class TextProcessor {
  private splitIntoChunks(text: string, batchSize: number): string[] {
    // Split by sentences (basic sentence detection)
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    
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

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      let retryCount = 0;
      let success = false;
      let processedText = "";

      // Try processing with one retry on failure
      while (retryCount < 2 && !success) {
        try {
          processedText = await llmService.processChunk({
            text: chunk,
            cleaningOptions: config.cleaningOptions,
            speakerConfig: config.speakerConfig,
            modelName: config.modelName,
            customInstructions: config.customInstructions,
          });

          // Validate output
          const validation = await llmService.validateOutput(
            chunk,
            processedText,
            config.modelName
          );

          if (validation.valid) {
            success = true;
            processedChunks.push(processedText);
            
            onProgress({
              chunkIndex: i,
              processedText,
              status: "success",
              retryCount,
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
        onProgress({
          chunkIndex: i,
          processedText: chunk,
          status: "failed",
          retryCount,
        });
      }
    }

    return processedChunks.join("\n\n");
  }
}

export const textProcessor = new TextProcessor();
