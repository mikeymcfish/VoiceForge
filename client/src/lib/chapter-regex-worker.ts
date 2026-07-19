import type {
  ChapterMarkerInsertion,
  ChapterRegexMatch,
} from "@shared/chapter-assist";

export interface ChapterRegexWorkerResult {
  matches: ChapterRegexMatch[];
  insertion?: ChapterMarkerInsertion;
}

interface ChapterRegexWorkerInput {
  operation: "preview" | "insert";
  text: string;
  pattern: string;
  caseSensitive: boolean;
}

type ChapterRegexWorkerResponse =
  | {
      id: string;
      ok: true;
      matches: ChapterRegexMatch[];
      insertion?: ChapterMarkerInsertion;
    }
  | {
      id: string;
      ok: false;
      error: string;
    };

export function runChapterRegexWorker(
  input: ChapterRegexWorkerInput,
  timeoutMs = 1_000
): Promise<ChapterRegexWorkerResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL("../workers/chapter-regex-worker.ts", import.meta.url),
      { type: "module" }
    );
    const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const timeout = window.setTimeout(() => {
      worker.terminate();
      reject(
        new Error(
          "The regex took too long to evaluate. Use a simpler, line-anchored pattern."
        )
      );
    }, timeoutMs);

    const finish = () => {
      window.clearTimeout(timeout);
      worker.terminate();
    };

    worker.onmessage = (event: MessageEvent<ChapterRegexWorkerResponse>) => {
      if (event.data.id !== requestId) return;
      finish();
      if (!event.data.ok) {
        reject(new Error(event.data.error));
        return;
      }
      resolve({
        matches: event.data.matches,
        insertion: event.data.insertion,
      });
    };
    worker.onerror = (event) => {
      finish();
      reject(new Error(event.message || "The chapter regex worker failed."));
    };
    worker.postMessage({ id: requestId, ...input });
  });
}
