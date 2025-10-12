import { HfInference } from "@huggingface/inference";
import type { CleaningOptions, SpeakerConfig } from "@shared/schema";

const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);

export interface ProcessChunkOptions {
  text: string;
  cleaningOptions: CleaningOptions;
  speakerConfig?: SpeakerConfig;
  modelName: string;
}

export class LLMService {
  private buildCleaningPrompt(text: string, options: CleaningOptions): string {
    const tasks: string[] = [];

    if (options.replaceSmartQuotes) {
      tasks.push("- Replace all smart quotes (", ", ', ') with standard ASCII quotes (\", ')");
    }
    if (options.fixOcrErrors) {
      tasks.push("- Fix OCR errors: correct spacing issues and merged words (e.g., 'thebook' → 'the book')");
    }
    if (options.correctSpelling) {
      tasks.push("- Correct common spelling mistakes and typos");
    }
    if (options.removeUrls) {
      tasks.push("- Remove all URLs and web links");
    }
    if (options.removeFootnotes) {
      tasks.push("- Remove footnote markers (numbers, asterisks) and extraneous metadata");
    }
    if (options.addPunctuation) {
      tasks.push("- Add appropriate punctuation after headers and loose numbers for better TTS prosody");
    }

    return `You are a text cleaning assistant for TTS (text-to-speech) preprocessing. Your task is to clean and repair the following text.

Apply these transformations:
${tasks.join("\n")}

Important rules:
- Preserve the original meaning and content
- Only fix errors, don't rewrite or rephrase
- Maintain paragraph structure
- Return ONLY the cleaned text, no explanations

Text to clean:
${text}

Cleaned text:`;
  }

  private buildSpeakerPrompt(
    text: string,
    config: SpeakerConfig
  ): string {
    const labelExample =
      config.labelFormat === "speaker"
        ? "Speaker 1:, Speaker 2:, etc."
        : "[1]:, [2]:, [3]:, etc.";

    if (config.mode === "format") {
      return `You are a dialogue formatting assistant. Convert the following text to a standardized multi-speaker format.

Requirements:
- Number of speakers: ${config.speakerCount}
- Label format: ${labelExample}
- Each speaker's line should start with their label
- Preserve all dialogue content exactly as written
- Only change the speaker label format

Text to format:
${text}

Formatted dialogue:`;
    } else {
      return `You are an intelligent dialogue parsing assistant. Analyze this prose text and extract dialogue between ${config.speakerCount} speakers.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each line as: ${labelExample} [dialogue text]
- Assign consistent speaker numbers based on who speaks
- Extract only the spoken dialogue, not narrative descriptions
- If a character name is detected, keep it consistent with one speaker number

Text to parse:
${text}

Structured dialogue:`;
    }
  }

  async processChunk(options: ProcessChunkOptions): Promise<string> {
    const { text, cleaningOptions, speakerConfig, modelName } = options;

    // Stage 1: Text cleaning
    const cleaningPrompt = this.buildCleaningPrompt(text, cleaningOptions);
    
    const stage1Response = await hf.chatCompletion({
      model: modelName,
      messages: [
        {
          role: "user",
          content: cleaningPrompt,
        },
      ],
      max_tokens: 2000,
      temperature: 0.3,
    });

    let processedText = stage1Response.choices[0]?.message?.content || text;

    // Stage 2: Speaker formatting (if configured)
    if (speakerConfig) {
      const speakerPrompt = this.buildSpeakerPrompt(processedText, speakerConfig);
      
      const stage2Response = await hf.chatCompletion({
        model: modelName,
        messages: [
          {
            role: "user",
            content: speakerPrompt,
          },
        ],
        max_tokens: 2000,
        temperature: 0.3,
      });

      processedText = stage2Response.choices[0]?.message?.content || processedText;
    }

    return processedText.trim();
  }

  async validateOutput(
    originalText: string,
    processedText: string,
    modelName: string
  ): Promise<{ valid: boolean; issues?: string[] }> {
    // Simple validation: check if output is not empty and has reasonable length
    if (!processedText || processedText.length < originalText.length * 0.5) {
      return {
        valid: false,
        issues: ["Output too short or empty"],
      };
    }

    // Check for common error patterns
    const issues: string[] = [];

    if (processedText.includes("[ERROR]") || processedText.includes("I cannot")) {
      issues.push("Model returned error message");
    }

    if (processedText.split("\n").length < 2 && originalText.split("\n").length > 5) {
      issues.push("Lost paragraph structure");
    }

    return {
      valid: issues.length === 0,
      issues: issues.length > 0 ? issues : undefined,
    };
  }
}

export const llmService = new LLMService();
