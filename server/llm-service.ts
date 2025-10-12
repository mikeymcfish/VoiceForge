import { HfInference } from "@huggingface/inference";
import type { CleaningOptions, SpeakerConfig } from "@shared/schema";

const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);

export interface ProcessChunkOptions {
  text: string;
  cleaningOptions: CleaningOptions;
  speakerConfig?: SpeakerConfig;
  modelName: string;
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
      prompt = `You are an intelligent dialogue parsing assistant. Analyze this prose text and extract dialogue between ${config.speakerCount} speakers.

Requirements:
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each line as: ${labelExample} [dialogue text]
- Assign consistent speaker numbers based on who speaks
- Extract only the spoken dialogue, not narrative descriptions
- If a character name is detected, keep it consistent with one speaker number`;
    }

    // Add character mapping if available
    if (config.characterMapping && config.characterMapping.length > 0) {
      const mappingList = config.characterMapping
        .map((char) => `  - ${char.name} = Speaker ${char.speakerNumber}`)
        .join("\n");
      
      prompt += `\n\nCharacter to Speaker Mapping (ONLY extract dialogue from these characters, use these exact assignments):
${mappingList}

IMPORTANT: Only extract dialogue from the characters listed above. Ignore dialogue from any other characters not in this mapping.`;
    }

    if (customInstructions) {
      prompt += `\n\nAdditional custom instructions:\n${customInstructions}`;
    }

    prompt += `\n\nText to format:\n${text}\n\nFormatted dialogue:`;

    return prompt;
  }

  async processChunk(options: ProcessChunkOptions): Promise<string> {
    const { text, cleaningOptions, speakerConfig, modelName, customInstructions } = options;

    // Stage 1: Text cleaning
    const cleaningPrompt = this.buildCleaningPrompt(text, cleaningOptions, customInstructions);
    
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

    // Stage 2: Speaker formatting (if configured and mode is not "none")
    if (speakerConfig && speakerConfig.mode !== "none") {
      const speakerPrompt = this.buildSpeakerPrompt(processedText, speakerConfig, customInstructions);
      
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
    modelName: string;
  }): Promise<Array<{ name: string; speakerNumber: number }>> {
    const { text, includeNarrator, modelName } = options;

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

    const content = response.choices[0]?.message?.content || "[]";
    
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
