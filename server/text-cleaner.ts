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
const MOJIBAKE_REPLACEMENTS: Record<string, string> = {
  "\u00e2\u20ac\u0153": "\"",
  "\u00e2\u20ac\u009d": "\"",
  "\u00e2\u20ac\u02dc": "'",
  "\u00e2\u20ac\u2122": "'",
  "\u00e2\u20ac\u201c": "-",
  "\u00e2\u20ac\u201d": "-",
  "\u00e2\u20ac\u00a6": "...",
  "\u00c2\u00ab": "\"",
  "\u00c2\u00bb": "\"",
};
const MOJIBAKE_RE = new RegExp(
  Object.keys(MOJIBAKE_REPLACEMENTS)
    .sort((left, right) => right.length - left.length)
    .map((value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|"),
  "g"
);
const NBSP_RE = /\u00a0/g;
const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>()]+/gi;
// Preserve bracket speaker labels such as `[1]:` while removing citation marks.
const BRACKET_REFERENCE_RE = /\[\s*(?:[0-9]+|[ivxlcdmIVXLCDM]+)(?:[\s,.;:-]*(?:[0-9]+|[ivxlcdmIVXLCDM]+))*\s*\](?!\s*:)/g;
const PAREN_FOOTNOTE_RE = /\(\s*(?:[0-9]+|[ivxlcdmIVXLCDM]+)(?:[\s,.;:-]*(?:[0-9]+|[ivxlcdmIVXLCDM]+))*\s*\)/g;
const CAMEL_CASE_SPLIT_RE = /([a-z])([A-Z][a-z]+)/g;
const MERGED_WORD_RE = /\b[a-zA-Z]{6,}\b/g;
const TERMINAL_TTS_PUNCTUATION_RE = /[.!?:;…]+(?:["'”’\])}]*)$/u;
const CHAPTER_MARKER_PREFIX_RE = /^\[chapter\]\s*/iu;
const STRUCTURAL_HEADING_RE = /^(?:(?:chapter|part|section|book|volume|act|scene|appendix)\s+(?:(?:\d+|[ivxlcdm]+|[a-z]+)(?:\s*[-–—:]\s*[\p{L}\p{N}][\p{L}\p{N}\s,'’&-]{0,80})?)|prologue|epilogue|preface|introduction|afterword|interlude)$/iu;
const BARE_SECTION_NUMBER_RE = /^(?:\d+|[ivxlcdm]+)$/iu;

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
    .replace(MOJIBAKE_RE, (value) => MOJIBAKE_REPLACEMENTS[value] ?? value)
    .replace(SMART_PUNCTUATION_RE, (char) => SMART_REPLACEMENTS[char] ?? char)
    .replace(NBSP_RE, " ");
}

function removeUrls(text: string): string {
  return text.replace(URL_RE, " ");
}

function removeReferences(text: string): string {
  return text.replace(BRACKET_REFERENCE_RE, " ").replace(PAREN_FOOTNOTE_RE, " ");
}

function isLikelyHeadingLine(line: string): boolean {
  const heading = line.trim().replace(CHAPTER_MARKER_PREFIX_RE, "").trim();
  if (!heading || TERMINAL_TTS_PUNCTUATION_RE.test(heading)) return false;
  if (STRUCTURAL_HEADING_RE.test(heading) || BARE_SECTION_NUMBER_RE.test(heading)) {
    return true;
  }
  const words = heading.split(/\s+/).filter(Boolean);
  return (
    words.length >= 2 &&
    words.length <= 12 &&
    heading.length <= 80 &&
    /^[A-Z0-9][A-Z0-9\s,'’&-]*$/u.test(heading)
  );
}

function fixHyphenationArtifacts(text: string): string {
  const withoutHyphenatedBreaks = text.replace(
    /(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])/g,
    ""
  );
  const lines = withoutHyphenatedBreaks.split("\n");
  const mergedLines: string[] = [];

  for (let index = 0; index < lines.length; index++) {
    let current = lines[index];
    while (index + 1 < lines.length) {
      const next = lines[index + 1];
      const nextTrimmed = next.trimStart();
      const isWrappedText = /[A-Za-z]$/u.test(current) && /^[A-Za-z]/u.test(nextTrimmed);
      if (
        !isWrappedText ||
        isLikelyHeadingLine(current) ||
        isLikelyHeadingLine(next)
      ) {
        break;
      }
      current += ` ${nextTrimmed}`;
      index++;
    }
    mergedLines.push(current);
  }
  return mergedLines.join("\n");
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

function addTtsHeadingPunctuation(text: string): string {
  const lines = text.split("\n");
  return lines
    .map((line, index) => {
      const match = /^(\s*)(.*?)(\s*)$/u.exec(line);
      if (!match) return line;
      const [, leading, content, trailing] = match;
      if (!content || TERMINAL_TTS_PUNCTUATION_RE.test(content)) return line;

      const heading = content.replace(CHAPTER_MARKER_PREFIX_RE, "").trim();
      const isolated =
        (index === 0 || !lines[index - 1]?.trim()) &&
        (index === lines.length - 1 || !lines[index + 1]?.trim());
      const isStructuralHeading = STRUCTURAL_HEADING_RE.test(heading);
      const isLooseSectionNumber = isolated && BARE_SECTION_NUMBER_RE.test(heading);
      if (!isStructuralHeading && !isLooseSectionNumber) return line;
      return `${leading}${content}:${trailing}`;
    })
    .join("\n");
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

  // This is a reliable final safeguard for the explicitly enabled TTS option.
  // The LLM still handles contextual punctuation; this covers clear structural
  // headings if the model leaves one unchanged.
  if (options.addPunctuation && phase === "post") {
    const punctuated = addTtsHeadingPunctuation(text);
    if (punctuated !== text) {
      applied.push("addPunctuation");
      text = punctuated;
    }
  }

  const normalized = normalizeSpacing(text);
  if (normalized !== text) applied.push("normalizeSpacing");
  text = normalized;

  return { text, applied };
}
