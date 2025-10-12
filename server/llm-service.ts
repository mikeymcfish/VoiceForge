import { HfInference } from "@huggingface/inference";
import type { CleaningOptions, SpeakerConfig, ModelSource } from "@shared/schema";
import { LocalModelService } from "./local-model-service";

// Check if HuggingFace API token is available
const apiToken = process.env.HUGGINGFACE_API_TOKEN;
if (!apiToken) {
  console.warn("⚠️  HUGGINGFACE_API_TOKEN not found in environment variables!");
  console.warn("   API mode will not work. Please either:");
  console.warn("   1. Add HUGGINGFACE_API_TOKEN to Replit Secrets");
  console.warn("   2. Use Local Models instead (no API token required)");
}

const hf = new HfInference(apiToken);

export interface ProcessChunkOptions {
  text: string;
  cleaningOptions: CleaningOptions;
  speakerConfig?: SpeakerConfig;
  modelSource?: ModelSource;
  modelName: string; // API model name
  localModelName?: string; // Local model name
  customInstructions?: string;
}

export class LLMService {
  private buildCleaningPrompt(
    text: string, 
    options: CleaningOptions,
    customInstructions?: string
  ): string {
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

    let prompt = `You are a text cleaning assistant for TTS (text-to-speech) preprocessing. Your task is to clean and repair the following text.

Apply these transformations:
${tasks.join("\n")}

Important rules:
- Preserve the original meaning and content
- Only fix errors, don't rewrite or rephrase
- Maintain paragraph structure
- Return ONLY the cleaned text, no explanations`;

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    prompt += `\n\nText to clean:\n${text}\n\nCleaned text:`;

    return prompt;
  }

  private buildSpeakerPrompt(
    text: string,
    config: SpeakerConfig,
    customInstructions?: string
  ): string {
    const labelExample =
      config.labelFormat === "speaker"
        ? "Speaker 1:, Speaker 2:, etc."
        : "[1]:, [2]:, [3]:, etc.";

    let prompt = "";

    if (config.mode === "format") {
      prompt = `You are a dialogue formatting assistant. Convert the following text to a standardized multi-speaker format.

Requirements:
- Number of speakers: ${config.speakerCount}
- Label format: ${labelExample}
- Each speaker's line should start with their label
- Preserve all dialogue content exactly as written
- Only change the speaker label format`;
    } else {
      // Check if narrator is in the character mapping
      const hasNarrator = config.characterMapping?.some(
        (char) => char.name.toLowerCase() === "narrator"
      );

      if (hasNarrator) {
        prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and structure it for multi-speaker TTS.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each speaking character's line as: ${labelExample} [dialogue text]
- Remove dialogue attribution tags (e.g., "he said", "she replied", "Bob asked")
- Preserve narrative descriptions and non-dialogue text as Narrator lines
- Assign consistent speaker numbers based on who speaks`;
      } else {
        prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and extract dialogue between ${config.speakerCount} speakers.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each line as: ${labelExample} [dialogue text]
- Assign consistent speaker numbers based on who speaks
- Extract only the spoken dialogue, not narrative descriptions
- Remove dialogue attribution tags (e.g., "he said", "she replied")
- If a character name is detected, keep it consistent with one speaker number`;
      }
    }

    // Add character mapping if available
    if (config.characterMapping && config.characterMapping.length > 0) {
      const mappingList = config.characterMapping
        .map((char) => `  - ${char.name} = Speaker ${char.speakerNumber}`)
        .join("\n");
      
      const hasNarrator = config.characterMapping.some(
        (char) => char.name.toLowerCase() === "narrator"
      );

      if (hasNarrator) {
        prompt += `\n\nCharacter to Speaker Mapping (use these exact assignments):
${mappingList}

IMPORTANT: 
- Only extract dialogue from the speaking characters listed above
- Assign narrative descriptions and non-dialogue portions to the Narrator
- Remove dialogue attribution tags like "he said", "she whispered", "Alice asked"
- Ignore dialogue from any characters not in this mapping`;
      } else {
        prompt += `\n\nCharacter to Speaker Mapping (ONLY extract dialogue from these characters, use these exact assignments):
${mappingList}

IMPORTANT: Only extract dialogue from the characters listed above. Ignore dialogue from any other characters not in this mapping.`;
      }
    }

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    prompt += `\n\nText to format:\n${text}\n\nFormatted dialogue:`;

    return prompt;
  }

  private async runInference(prompt: string, options: ProcessChunkOptions): Promise<string> {
    const { modelSource = 'api', modelName, localModelName } = options;

    if (modelSource === 'local') {
      if (!localModelName) {
        throw new Error('Local model name is required for local inference');
      }
      return await LocalModelService.generateText(localModelName, prompt);
    } else {
      // Check if API token is available
      if (!apiToken) {
        throw new Error(
          'HuggingFace API token not found. Please add HUGGINGFACE_API_TOKEN to your Replit Secrets, or switch to Local Models (no API token required).'
        );
      }

      try {
        const response = await hf.chatCompletion({
          model: modelName,
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
          max_tokens: 2000,
          temperature: 0.3,
        });

        return response.choices[0]?.message?.content || '';
      } catch (error) {
        // Provide more helpful error messages
        if (error instanceof Error) {
          if (error.message.includes('401') || error.message.includes('unauthorized') || 
              error.message.includes('authentication') || error.message.includes('token')) {
            throw new Error(
              'HuggingFace API authentication failed. Your API token may be invalid or expired. Please check your HUGGINGFACE_API_TOKEN in Replit Secrets, or switch to Local Models.'
            );
          }
        }
        throw error;
      }
    }
  }

  async processChunk(options: ProcessChunkOptions): Promise<string> {
    const { text, cleaningOptions, speakerConfig, customInstructions } = options;

    // Stage 1: Text cleaning
    const cleaningPrompt = this.buildCleaningPrompt(text, cleaningOptions, customInstructions);
    let processedText = await this.runInference(cleaningPrompt, options);

    if (!processedText) {
      processedText = text;
    }

    // Stage 2: Speaker formatting (if configured and mode is not "none")
    if (speakerConfig && speakerConfig.mode !== "none") {
      const speakerPrompt = this.buildSpeakerPrompt(processedText, speakerConfig, customInstructions);
      const stage2Text = await this.runInference(speakerPrompt, options);
      
      if (stage2Text) {
        processedText = stage2Text;
      }
    }

    return processedText.trim();
  }

  async validateOutput(
    originalText: string,
    processedText: string,
    modelSource?: ModelSource
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

    // For local models, use lightweight validation only (no API calls)
    // For API models, we could add more sophisticated validation if needed
    
    return {
      valid: issues.length === 0,
      issues: issues.length > 0 ? issues : undefined,
    };
  }

  getPromptPreviews(
    sampleText: string,
    cleaningOptions: CleaningOptions,
    speakerConfig?: SpeakerConfig,
    customInstructions?: string
  ): { stage1: string; stage2?: string } {
    const stage1 = this.buildCleaningPrompt(sampleText, cleaningOptions, customInstructions);
    
    const result: { stage1: string; stage2?: string } = { stage1 };

    if (speakerConfig && speakerConfig.mode !== "none") {
      result.stage2 = this.buildSpeakerPrompt(sampleText, speakerConfig, customInstructions);
    }

    return result;
  }

  async extractCharacters(options: {
    text: string;
    includeNarrator: boolean;
    modelSource?: ModelSource;
    modelName: string;
    localModelName?: string;
  }): Promise<Array<{ name: string; speakerNumber: number }>> {
    const { text, includeNarrator, modelSource = 'api', modelName, localModelName } = options;

    const prompt = `You are a character extraction assistant for multi-speaker TTS systems. Analyze the following text sample and extract all character/speaker names that appear.

Requirements:
- Extract only actual character names that speak in the text
- List each unique character only once
- Return names in order of first appearance
${includeNarrator ? "- Include 'Narrator' as a character if there is narrative/descriptive text" : "- Do NOT include narrator or descriptive text"}
- Return ONLY a JSON array of character names, no explanations

Format your response as a JSON array:
["Character1", "Character2", "Character3"]

Text sample:
${text}

Character names (JSON array only):`;

    let content: string;

    if (modelSource === 'local') {
      if (!localModelName) {
        throw new Error('Local model name is required for local inference');
      }
      content = await LocalModelService.generateText(localModelName, prompt);
    } else {
      // Check if API token is available
      if (!apiToken) {
        throw new Error(
          'HuggingFace API token not found. Please add HUGGINGFACE_API_TOKEN to your Replit Secrets, or switch to Local Models (no API token required).'
        );
      }

      try {
        const response = await hf.chatCompletion({
          model: modelName,
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
          max_tokens: 500,
          temperature: 0.2,
        });
        content = response.choices[0]?.message?.content || "[]";
      } catch (error) {
        // Provide more helpful error messages
        if (error instanceof Error) {
          if (error.message.includes('401') || error.message.includes('unauthorized') || 
              error.message.includes('authentication') || error.message.includes('token')) {
            throw new Error(
              'HuggingFace API authentication failed. Your API token may be invalid or expired. Please check your HUGGINGFACE_API_TOKEN in Replit Secrets, or switch to Local Models.'
            );
          }
        }
        throw error;
      }
    }
    
    // Extract JSON array from response
    let characterNames: string[] = [];
    try {
      // Try to parse directly
      characterNames = JSON.parse(content.trim());
    } catch {
      // Try to extract JSON array from markdown code block or other formatting
      const jsonMatch = content.match(/\[[\s\S]*?\]/);
      if (jsonMatch) {
        try {
          characterNames = JSON.parse(jsonMatch[0]);
        } catch {
          console.error("Failed to parse character names from LLM response:", content);
        }
      }
    }

    // Ensure we have an array
    if (!Array.isArray(characterNames)) {
      characterNames = [];
    }

    // Map character names to speaker numbers (1-indexed)
    return characterNames.map((name, index) => ({
      name: String(name).trim(),
      speakerNumber: index + 1,
    }));
  }
}

export const llmService = new LLMService();
