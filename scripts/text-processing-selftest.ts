import assert from "node:assert/strict";
import type { CleaningOptions } from "@shared/schema";
import {
  chunkTextBySentences,
  chunkTextPreservingStructure,
  countWords,
  segmentSentences,
} from "@shared/text-utils";
import { applyDeterministicCleaning } from "../server/text-cleaner";
import { buildOllamaOptions, isThinkingOllamaModel } from "../shared/model-utils";

const allLocalOptions: CleaningOptions = {
  replaceSmartQuotes: true,
  fixOcrErrors: false,
  correctSpelling: false,
  removeUrls: true,
  removeFootnotes: true,
  addPunctuation: false,
  insertChapterBreaks: false,
  fixHyphenation: true,
};

const cleaned = applyDeterministicCleaning(
  '“Hello,” she said… Visit https://example.com.\n\n[12] Keep [1]: dialogue. A hy-\nphen.',
  allLocalOptions
);
assert.equal(cleaned.text, '"Hello," she said... Visit\n\nKeep [1]: dialogue. A hyphen.');
assert.ok(cleaned.applied.includes("replaceSmartQuotes"));
assert.ok(cleaned.applied.includes("removeUrls"));
assert.ok(cleaned.applied.includes("removeFootnotes"));
assert.ok(cleaned.applied.includes("fixHyphenation"));

const ttsPunctuation = applyDeterministicCleaning(
  "Part 1\n\nThe journey begins.\n\nCHAPTER ONE\nThe next section starts here.",
  { ...allLocalOptions, addPunctuation: true },
  "post"
);
assert.equal(
  ttsPunctuation.text,
  "Part 1:\n\nThe journey begins.\n\nCHAPTER ONE:\nThe next section starts here."
);
assert.ok(ttsPunctuation.applied.includes("addPunctuation"));

assert.equal(isThinkingOllamaModel("qwen3:8b"), true);
assert.equal(isThinkingOllamaModel("llama3.1:8b"), false);
assert.equal(
  buildOllamaOptions({ temperature: 0.3, num_predict: 2_000 }, "ollama", "qwen3:8b").num_predict,
  4_096
);

const mojibake = applyDeterministicCleaning("\u00e2\u20ac\u0153Quoted\u00e2\u20ac\u009d", allLocalOptions);
assert.equal(mojibake.text, '"Quoted"');

const sentences = segmentSentences('Dr. Rivera arrived at 3.14 p.m. "Ready?" she asked. Yes.');
assert.equal(sentences.length, 3);
assert.equal(chunkTextBySentences(sentences.join(" "), 2).length, 2);
const structuredSource = "CHAPTER ONE\n\nFirst sentence.  Second sentence.\n\nThird paragraph.";
const structuredChunks = chunkTextPreservingStructure(structuredSource, 1);
assert.equal(
  structuredChunks.map((chunk) => chunk.text + chunk.separatorAfter).join(""),
  structuredSource
);
assert.equal(countWords("Don't split speaker-2's label"), 4);

console.log("Text processing self-test passed");
