import {
  findChapterRegexMatches,
  insertChapterMarkersFromRegex,
  type ChapterMarkerInsertion,
  type ChapterRegexMatch,
} from "@shared/chapter-assist";

interface ChapterRegexWorkerRequest {
  id: string;
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

const workerScope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<ChapterRegexWorkerRequest>) => void) | null;
  postMessage: (response: ChapterRegexWorkerResponse) => void;
};

workerScope.onmessage = (event) => {
  const request = event.data;
  try {
    if (request.operation === "insert") {
      const insertion = insertChapterMarkersFromRegex(
        request.text,
        request.pattern,
        request.caseSensitive
      );
      workerScope.postMessage({
        id: request.id,
        ok: true,
        matches: insertion.matches,
        insertion,
      });
      return;
    }

    workerScope.postMessage({
      id: request.id,
      ok: true,
      matches: findChapterRegexMatches(
        request.text,
        request.pattern,
        request.caseSensitive
      ),
    });
  } catch (error) {
    workerScope.postMessage({
      id: request.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

export {};
