import type { CleaningOptions } from "@shared/schema";
import fs from "fs";

const SMART_REPLACEMENTS: Record<string, string> = {
  "“": "\"",
  "”": "\"",
  "„": "\"",
  "«": "\"",
  "»": "\"",
  "‹": "\"",
  "›": "\"",
  "‘": "'",
  "’": "'",
  "‚": "'",
  "‛": "'",
  "–": "-",
  "—": "-",
  "−": "-",
  "‐": "-",
  "‑": "-",
  "‒": "-",
  "…": "...",
  "•": "-",
  "·": "-",
};

const SMART_PUNCTUATION_RE = /[“”„«»‹›‘’‚‛–—−‐‑‒…•·]/g;
const NBSP_RE = /\u00a0/g;
const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>()]+/gi;
const BRACKET_REFERENCE_RE = /\[\s*(?:[0-9]+|[ivxlcdmIVXLCDM]+)(?:[\s,.;:-]*(?:[0-9]+|[ivxlcdmIVXLCDM]+))*\s*\]/g;
const PAREN_FOOTNOTE_RE = /\(\s*(?:[0-9]+|[ivxlcdmIVXLCDM]+)(?:[\s,.;:-]*(?:[0-9]+|[ivxlcdmIVXLCDM]+))*\s*\)/g;
const CAMEL_CASE_SPLIT_RE = /([a-z])([A-Z][a-z]+)/g;
const MERGED_WORD_RE = /\b[a-zA-Z]{6,}\b/g;

const FALLBACK_WORDS: ReadonlyArray<string> = [
  "a", "able", "about", "above", "after", "again", "against", "air", "all", "along", "also",
  "always", "among", "an", "and", "another", "any", "anything", "are", "around", "as", "ask",
  "at", "away", "back", "be", "because", "become", "been", "before", "began", "behind", "being",
  "below", "between", "both", "brought", "but", "by", "call", "called", "can", "cannot", "come",
  "could", "day", "did", "didn", "do", "does", "done", "down", "each", "end", "enough", "even",
  "ever", "every", "face", "fact", "far", "feel", "felt", "few", "find", "first", "for", "form",
  "found", "from", "gave", "get", "give", "given", "go", "good", "got", "great", "had", "half",
  "hand", "has", "have", "having", "he", "head", "hear", "heard", "help", "her", "here", "high",
  "him", "his", "home", "house", "how", "however", "if", "in", "into", "is", "it", "its", "just",
  "keep", "knew", "know", "known", "land", "large", "last", "later", "least", "leave", "left",
  "let", "life", "like", "little", "long", "look", "looked", "made", "make", "man", "many",
  "may", "mean", "men", "might", "mind", "moment", "more", "most", "mother", "much", "must",
  "near", "need", "never", "new", "night", "no", "not", "nothing", "now", "of", "off", "old",
  "on", "once", "one", "only", "open", "or", "other", "our", "out", "over", "own", "part",
  "people", "place", "point", "put", "right", "room", "said", "same", "saw", "say", "says",
  "see", "seem", "seemed", "shall", "she", "should", "side", "since", "small", "so", "some",
  "someone", "something", "soon", "still", "such", "take", "taken", "tell", "than", "that",
  "the", "their", "them", "then", "there", "these", "they", "thing", "think", "this", "those",
  "though", "thought", "three", "through", "time", "to", "told", "too", "took", "toward",
  "turn", "two", "under", "up", "upon", "us", "use", "very", "want", "was", "way", "we", "well",
  "went", "were", "what", "when", "where", "which", "while", "who", "whole", "why", "will",
  "with", "within", "without", "word", "words", "work", "would", "year", "years", "yes", "you",
  "young", "your"
];

const DICT_CANDIDATE_PATHS = [
  "/usr/share/dict/words",
  "/usr/share/dict/american-english",
  "/usr/share/dict/english",
  "/usr/share/dict/british-english",
];

const MAX_LEXICON_SIZE = 60000;

let cachedLexicon: Set<string> | null = null;

function loadLexicon(): Set<string> {
  if (cachedLexicon) return cachedLexicon;
  const lexicon = new Set<string>(FALLBACK_WORDS);
  for (const candidate of DICT_CANDIDATE_PATHS) {
    try {
      if (!fs.existsSync(candidate)) continue;
      const contents = fs.readFileSync(candidate, "utf-8");
      for (const line of contents.split(/\r?\n/)) {
        const word = line.trim().toLowerCase();
        if (!word) continue;
        if (!/^[a-z]+$/.test(word)) continue;
        lexicon.add(word);
        if (lexicon.size >= MAX_LEXICON_SIZE) break;
      }
    } catch {
      // Ignore dictionary read errors and proceed with current lexicon.
    }
    if (lexicon.size >= MAX_LEXICON_SIZE) break;
  }
  cachedLexicon = lexicon;
  return lexicon;
}

function replaceSmartPunctuation(text: string): string {
  return text
    .replace(SMART_PUNCTUATION_RE, (char) => SMART_REPLACEMENTS[char] ?? char)
    .replace(NBSP_RE, " ");
}

function removeUrls(text: string): string {
  return text.replace(URL_RE, " ");
}

function removeReferences(text: string): string {
  return text.replace(BRACKET_REFERENCE_RE, " ").replace(PAREN_FOOTNOTE_RE, " ");
}

function fixHyphenationArtifacts(text: string): string {
  let result = text;
  result = result.replace(/(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])/g, "");
  result = result.replace(/(?<=[A-Za-z])\n(?=[A-Za-z])/g, " ");
  return result;
}

function splitCamelCase(text: string): string {
  return text.replace(CAMEL_CASE_SPLIT_RE, "$1 $2");
}

function findLexiconSplit(original: string, lexicon: Set<string>, depth = 0): string[] | null {
  const lower = original.toLowerCase();
  if (lexicon.has(lower) || depth > 3 || lower.length < 6) {
    return null;
  }
  const minPart = 3;
  for (let i = lower.length - minPart; i >= minPart; i--) {
    const leftLower = lower.slice(0, i);
    const rightLower = lower.slice(i);
    if (!lexicon.has(leftLower)) continue;
    const rightOriginal = original.slice(i);
    if (lexicon.has(rightLower)) {
      return [original.slice(0, i), rightOriginal];
    }
    const deeper = findLexiconSplit(rightOriginal, lexicon, depth + 1);
    if (deeper) {
      return [original.slice(0, i), ...deeper];
    }
  }
  return null;
}

function splitMergedWords(text: string, lexicon: Set<string>): string {
  return text.replace(MERGED_WORD_RE, (word) => {
    if (word.length > 30) return word;
    const lower = word.toLowerCase();
    if (lexicon.has(lower)) return word;
    const split = findLexiconSplit(word, lexicon);
    return split ? split.join(" ") : word;
  });
}

function normalizeSpacing(text: string): string {
  return text
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export interface DeterministicCleaningResult {
  text: string;
  applied: string[];
}

export function applyDeterministicCleaning(
  input: string,
  options: CleaningOptions,
  phase: "pre" | "post" = "pre"
): DeterministicCleaningResult {
  let text = input;
  const applied: string[] = [];

  if (options.replaceSmartQuotes) {
    const replaced = replaceSmartPunctuation(text);
    if (replaced !== text) {
      applied.push("replaceSmartQuotes");
      text = replaced;
    }
  }

  if (options.removeUrls) {
    const stripped = removeUrls(text);
    if (stripped !== text) {
      applied.push("removeUrls");
      text = stripped;
    }
  }

  if (options.removeFootnotes && phase === "pre") {
    const scrubbed = removeReferences(text);
    if (scrubbed !== text) {
      applied.push("removeFootnotes");
      text = scrubbed;
    }
  }

  if (options.fixHyphenation) {
    const merged = fixHyphenationArtifacts(text);
    if (merged !== text) {
      applied.push("fixHyphenation");
      text = merged;
    }
  }

  if (options.fixOcrErrors) {
    const camelSplit = splitCamelCase(text);
    if (camelSplit !== text) {
      applied.push("splitCamelCase");
      text = camelSplit;
    }
    const lexicon = loadLexicon();
    const mergedSplit = splitMergedWords(text, lexicon);
    if (mergedSplit !== text) {
      applied.push("splitMergedWords");
      text = mergedSplit;
    }
  }

  text = normalizeSpacing(text);

  return { text, applied };
}
