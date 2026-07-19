import assert from "node:assert/strict";
import {
  buildChapterRegexPattern,
  findChapterRegexMatches,
  insertChapterMarkersFromRegex,
} from "../shared/chapter-assist";

function testMonthDayPreset(): void {
  const pattern = buildChapterRegexPattern("month-day-heading");
  const source = [
    "January 2",
    "Morning narration.",
    "jan 3rd",
    "September 14, 2026 — Evening",
    "We met again on January 5.",
    "NotAMonth 6",
  ].join("\n");

  assert.deepEqual(
    findChapterRegexMatches(source, pattern).map((match) => match.text),
    ["January 2", "jan 3rd", "September 14, 2026 — Evening"]
  );
  assert.deepEqual(
    findChapterRegexMatches(source, pattern, true).map((match) => match.text),
    ["January 2", "September 14, 2026 — Evening"]
  );
}

function testOtherPresets(): void {
  assert.equal(
    findChapterRegexMatches(
      "Chapter 12: Arrival\nbody",
      buildChapterRegexPattern("chapter-number")
    ).length,
    1
  );
  assert.equal(
    findChapterRegexMatches(
      "Appendix. 7\nAppendixx 8",
      buildChapterRegexPattern("custom-word-number", "Appendix.")
    ).length,
    1
  );
  assert.equal(
    findChapterRegexMatches(
      "PART IV\nSeptember 3",
      buildChapterRegexPattern("common-headings")
    ).length,
    2
  );
}

function testInsertionPreservesTextAndNewlines(): void {
  const pattern = buildChapterRegexPattern("month-day-heading");
  const source = "Preface\r\n  January 2\r\nText\r\nJanuary 3\r\n";
  const result = insertChapterMarkersFromRegex(source, pattern);

  assert.equal(
    result.text,
    "Preface\r\n  [CHAPTER] January 2\r\nText\r\n[CHAPTER] January 3\r\n"
  );
  assert.equal(result.insertedCount, 2);
  assert.equal(result.skippedExistingCount, 0);
  assert.equal(result.matches[0].lineNumber, 2);
  assert.equal(result.matches[1].lineNumber, 4);
}

function testExistingMarkersAreNotDuplicated(): void {
  const pattern = String.raw`^\s*.*January\s+\d{1,2}.*$`;
  const source = "[CHAPTER] January 2\nJanuary 3\n[CHAPTER]\nJanuary 4";
  const result = insertChapterMarkersFromRegex(source, pattern);

  assert.equal(
    result.text,
    "[CHAPTER] January 2\n[CHAPTER] January 3\n[CHAPTER]\nJanuary 4"
  );
  assert.equal(result.insertedCount, 1);
  assert.equal(result.skippedExistingCount, 2);
  assert.equal(result.matches[0].alreadyMarked, true);
  assert.equal(result.matches[2].alreadyMarked, true);
}

function testValidation(): void {
  assert.throws(
    () => findChapterRegexMatches("January 2", "["),
    /Invalid regex/
  );
  assert.throws(
    () => findChapterRegexMatches("January 2", ".*"),
    /must not match an empty line/
  );
  assert.throws(
    () => findChapterRegexMatches("January 2", ""),
    /Enter a regex pattern/
  );
}

testMonthDayPreset();
testOtherPresets();
testInsertionPreservesTextAndNewlines();
testExistingMarkersAreNotDuplicated();
testValidation();

console.log("chapter assist self-test passed");
