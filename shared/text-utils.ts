const FALLBACK_SENTENCE_RE = /[^.!?\n]+(?:[.!?]+["'\u201d\u2019)\]]*|(?=\n|$))/g;
const LEADING_ABBREVIATION_RE = /^(?:(?:mr|mrs|ms|dr|prof|sr|jr|st|mt|gen|col|capt|lt|sgt|rev|hon|pres|gov|sen|rep|vs|etc)|[a-z])\.$/i;
const ATTRIBUTION_RE = /^(?:he|she|they|i|[\p{Lu}][\p{L}'-]+)\s+(?:said|asked|replied|whispered|shouted|called|answered|added|continued|murmured|cried|yelled|wondered)\b/iu;

interface SentenceSpan {
  start: number;
  end: number;
  text: string;
}

export interface StructuredTextChunk {
  text: string;
  separatorAfter: string;
}

function mergeSentenceFragments(segments: SentenceSpan[], source: string): SentenceSpan[] {
  const merged: SentenceSpan[] = [];
  for (const segment of segments) {
    const previous = merged.at(-1);
    if (
      previous &&
      (LEADING_ABBREVIATION_RE.test(previous.text) ||
        (/\?["'\u201d\u2019]?\s*$/.test(previous.text) && ATTRIBUTION_RE.test(segment.text)))
    ) {
      const combined = {
        start: previous.start,
        end: segment.end,
        text: source.slice(previous.start, segment.end).trim(),
      };
      merged[merged.length - 1] = combined;
    } else {
      merged.push(segment);
    }
  }
  return merged;
}

function createSentenceSpans(source: string): SentenceSpan[] {
  const spans: SentenceSpan[] = [];

  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });
    for (const item of segmenter.segment(source)) {
      const raw = item.segment;
      const leading = raw.search(/\S/);
      if (leading < 0) continue;
      const trailing = raw.search(/\s+$/);
      const start = item.index + leading;
      const end = trailing >= 0 ? item.index + trailing : item.index + raw.length;
      const text = source.slice(start, end).trim();
      if (text) spans.push({ start, end, text });
    }
  } else {
    for (const match of source.matchAll(new RegExp(FALLBACK_SENTENCE_RE.source, "g"))) {
      const raw = match[0];
      const start = match.index ?? 0;
      const text = raw.trim();
      if (!text) continue;
      const leading = raw.indexOf(text);
      spans.push({ start: start + leading, end: start + leading + text.length, text });
    }
  }

  if (spans.length === 0 && source.trim()) {
    const start = source.search(/\S/);
    const end = source.search(/\s+$/);
    spans.push({
      start: Math.max(0, start),
      end: end >= 0 ? end : source.length,
      text: source.trim(),
    });
  }

  return mergeSentenceFragments(spans, source);
}

/**
 * Segment prose into sentences without breaking common abbreviations, decimals,
 * or punctuation followed by closing quotes. Modern runtimes use the built-in
 * Unicode-aware segmenter; the fallback keeps the app usable in older engines.
 */
export function segmentSentences(text: string): string[] {
  const normalized = text.replace(/\r\n?/g, "\n").trim();
  if (!normalized) return [];
  return createSentenceSpans(normalized).map((span) => span.text);
}

export function chunkTextPreservingStructure(text: string, batchSize: number): StructuredTextChunk[] {
  const normalized = text.replace(/\r\n?/g, "\n").trim();
  if (!normalized) return [];
  const spans = createSentenceSpans(normalized);
  const safeBatchSize = Math.max(1, Math.floor(batchSize) || 1);
  const chunks: StructuredTextChunk[] = [];

  for (let index = 0; index < spans.length; index += safeBatchSize) {
    const group = spans.slice(index, index + safeBatchSize);
    const next = spans[index + safeBatchSize];
    const start = group[0].start;
    const end = group.at(-1)!.end;
    const text = normalized.slice(start, end);
    const separatorAfter = normalized.slice(end, next?.start ?? normalized.length);
    if (text.trim()) chunks.push({ text, separatorAfter });
  }

  return chunks;
}

export function chunkTextBySentences(text: string, batchSize: number): string[] {
  return chunkTextPreservingStructure(text, batchSize).map((chunk) => chunk.text);
}

export function countWords(text: string): number {
  const matches = text.trim().match(/[\p{L}\p{N}]+(?:[\u2019'-][\p{L}\p{N}]+)*/gu);
  return matches?.length ?? 0;
}
