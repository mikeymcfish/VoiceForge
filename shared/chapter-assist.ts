export type ChapterRegexPreset =
  | "common-headings"
  | "chapter-number"
  | "part-number"
  | "month-day-heading"
  | "roman-numerals"
  | "custom-word-number"
  | "custom-regex";

export interface ChapterRegexMatch {
  lineNumber: number;
  text: string;
  alreadyMarked: boolean;
}

export interface ChapterMarkerInsertion {
  text: string;
  matches: ChapterRegexMatch[];
  insertedCount: number;
  skippedExistingCount: number;
}

const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Sept",
  "Oct",
  "Nov",
  "Dec",
].join("|");

const CHAPTER_MARKER_AT_LINE_START = /^\s*\[chapter\](?:\s|$)/i;
const STANDALONE_CHAPTER_MARKER = /^\s*\[chapter\]\s*$/i;

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function buildChapterRegexPattern(
  preset: ChapterRegexPreset,
  customHeadingWord = "Section"
): string {
  const numberedHeading = String.raw`(?:\d+|[ivxlcdm]+)\b`;
  const monthDay = String.raw`(?:${MONTH_NAMES})\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?\b`;

  switch (preset) {
    case "chapter-number":
      return String.raw`^\s*chapter\s+${numberedHeading}.*$`;
    case "part-number":
      return String.raw`^\s*part\s+${numberedHeading}.*$`;
    case "month-day-heading":
      return String.raw`^\s*${monthDay}.*$`;
    case "roman-numerals":
      return String.raw`^\s*(?:chapter\s+)?[ivxlcdm]+\b.*$`;
    case "common-headings":
      return String.raw`^\s*(?:chapter\s+${numberedHeading}|part\s+${numberedHeading}|${monthDay}).*$`;
    case "custom-word-number": {
      const token = escapeRegex(customHeadingWord.trim() || "Section");
      return String.raw`^\s*${token}\s+${numberedHeading}.*$`;
    }
    case "custom-regex":
      return "";
  }
}

export function compileChapterRegex(pattern: string, caseSensitive: boolean): RegExp {
  const normalized = pattern.trim();
  if (!normalized) throw new Error("Enter a regex pattern.");
  if (normalized.length > 1_000) {
    throw new Error("Regex patterns may contain at most 1,000 characters.");
  }

  let regex: RegExp;
  try {
    regex = new RegExp(normalized, caseSensitive ? "" : "i");
  } catch (error) {
    throw new Error(
      `Invalid regex: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  if (regex.test("")) {
    throw new Error("The regex must not match an empty line.");
  }
  return regex;
}

function lineEntries(text: string): Array<{ line: string; separator: string }> {
  const parts = text.split(/(\r\n|\n|\r)/);
  const entries: Array<{ line: string; separator: string }> = [];
  for (let index = 0; index < parts.length; index += 2) {
    entries.push({
      line: parts[index] ?? "",
      separator: parts[index + 1] ?? "",
    });
  }
  return entries;
}

export function findChapterRegexMatches(
  text: string,
  pattern: string,
  caseSensitive = false
): ChapterRegexMatch[] {
  const regex = compileChapterRegex(pattern, caseSensitive);
  const matches: ChapterRegexMatch[] = [];
  const entries = lineEntries(text);
  for (const [index, entry] of entries.entries()) {
    if (!entry.line.trim() || !regex.test(entry.line)) continue;
    matches.push({
      lineNumber: index + 1,
      text: entry.line,
      alreadyMarked:
        CHAPTER_MARKER_AT_LINE_START.test(entry.line) ||
        (index > 0 && STANDALONE_CHAPTER_MARKER.test(entries[index - 1].line)),
    });
  }
  return matches;
}

export function insertChapterMarkersFromRegex(
  text: string,
  pattern: string,
  caseSensitive = false
): ChapterMarkerInsertion {
  const regex = compileChapterRegex(pattern, caseSensitive);
  const matches: ChapterRegexMatch[] = [];
  let insertedCount = 0;
  let skippedExistingCount = 0;
  const entries = lineEntries(text);

  const updatedText = entries
    .map((entry, index) => {
      if (!entry.line.trim() || !regex.test(entry.line)) {
        return `${entry.line}${entry.separator}`;
      }

      const alreadyMarked =
        CHAPTER_MARKER_AT_LINE_START.test(entry.line) ||
        (index > 0 && STANDALONE_CHAPTER_MARKER.test(entries[index - 1].line));
      matches.push({
        lineNumber: index + 1,
        text: entry.line,
        alreadyMarked,
      });
      if (alreadyMarked) {
        skippedExistingCount += 1;
        return `${entry.line}${entry.separator}`;
      }

      const content = entry.line.trimStart();
      const indentation = entry.line.slice(0, entry.line.length - content.length);
      insertedCount += 1;
      return `${indentation}[CHAPTER] ${content}${entry.separator}`;
    })
    .join("");

  return {
    text: updatedText,
    matches,
    insertedCount,
    skippedExistingCount,
  };
}
