import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js
env.cacheDir = './.cache/huggingface';

// Available local models (ONNX format)
export const AVAILABLE_LOCAL_MODELS = [
  {
    id: 'Xenova/LaMini-Flan-T5-783M',
    name: 'LaMini-Flan-T5-783M',
    description: 'Text generation model (783M params, good for text processing)',
    size: '~800MB',
    task: 'text2text-generation',
  },
  {
    id: 'Xenova/flan-t5-small',
    name: 'Flan-T5 Small',
    description: 'Smaller text generation model (80M params, faster)',
    size: '~300MB',
    task: 'text2text-generation',
  },
  {
    id: 'Xenova/flan-t5-base',
    name: 'Flan-T5 Base',
    description: 'Base text generation model (250M params, balanced)',
    size: '~500MB',
    task: 'text2text-generation',
  },
] as const;

export class LocalModelService {
  private static instances: Map<string, any> = new Map();
  private static downloadProgress: Map<string, number> = new Map();

  static async getModel(modelId: string): Promise<any> {
    // Return cached instance if available
    if (this.instances.has(modelId)) {
      return this.instances.get(modelId)!;
    }

    // Find model config
    const modelConfig = AVAILABLE_LOCAL_MODELS.find(m => m.id === modelId);
    if (!modelConfig) {
      throw new Error(`Model ${modelId} not found in available models`);
    }

    console.log(`Loading local model: ${modelId}...`);

    // Create pipeline with progress callback
    const pipe = await pipeline(
      modelConfig.task,
      modelId,
      {
        progress_callback: (progress: any) => {
          if (progress.status === 'progress' && progress.file) {
            const percent = Math.round((progress.loaded / progress.total) * 100);
            this.downloadProgress.set(modelId, percent);
            console.log(`Downloading ${progress.file}: ${percent}%`);
          }
        }
      }
    );

    // Cache the instance
    this.instances.set(modelId, pipe);
    console.log(`Model ${modelId} loaded successfully`);

    return pipe;
  }

  static async generateText(modelId: string, prompt: string): Promise<string> {
    const model = await this.getModel(modelId);
    
    // Generate text
    const result = await model(prompt, {
      max_new_tokens: 512,
      temperature: 0.3,
      top_p: 0.9,
      repetition_penalty: 1.1,
    });

    // Extract generated text from result
    if (Array.isArray(result) && result[0]?.generated_text) {
      return result[0].generated_text;
    } else if (typeof result === 'object' && 'generated_text' in result) {
      return (result as any).generated_text;
    } else if (Array.isArray(result) && result[0]?.text) {
      return result[0].text;
    }

    return String(result);
  }

  static getDownloadProgress(modelId: string): number {
    return this.downloadProgress.get(modelId) || 0;
  }

  static isModelLoaded(modelId: string): boolean {
    return this.instances.has(modelId);
  }

  static clearCache(modelId?: string) {
    if (modelId) {
      this.instances.delete(modelId);
      this.downloadProgress.delete(modelId);
    } else {
      this.instances.clear();
      this.downloadProgress.clear();
    }
  }
}
