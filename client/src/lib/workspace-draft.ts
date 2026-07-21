export type PrepareWorkspaceDraft = {
  source: string;
  output: string;
  updatedAt: number;
};

export type TtsWorkspaceDraft = {
  indexText: string;
  vibeText: string;
  updatedAt: number;
};

type StoredDraft<T> = T & { key: string };

const DATABASE_NAME = "voiceforge-workspace";
const DATABASE_VERSION = 1;
const STORE_NAME = "drafts";
const PREPARE_DRAFT_KEY = "prepare-v1";
const TTS_HANDOFF_KEY = "tts-handoff-v1";
const TTS_WORKSPACE_KEY = "tts-workspace-v1";

function databaseAvailable(): boolean {
  return typeof window !== "undefined" && "indexedDB" in window;
}

function openDatabase(): Promise<IDBDatabase> {
  if (!databaseAvailable()) {
    return Promise.reject(new Error("IndexedDB is unavailable in this browser."));
  }
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open the workspace database."));
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Workspace storage request failed."));
  });
}

function transactionComplete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Workspace storage transaction failed."));
    transaction.onabort = () => reject(transaction.error ?? new Error("Workspace storage transaction was aborted."));
  });
}

async function readDraft<T>(key: string): Promise<T | null> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const record = await requestResult(transaction.objectStore(STORE_NAME).get(key) as IDBRequest<StoredDraft<T> | undefined>);
    await transactionComplete(transaction);
    if (!record) return null;
    const { key: _key, ...draft } = record;
    return draft as T;
  } finally {
    database.close();
  }
}

async function writeDraft<T>(key: string, draft: T): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    await requestResult(transaction.objectStore(STORE_NAME).put({ key, ...draft } as StoredDraft<T>));
    await transactionComplete(transaction);
  } finally {
    database.close();
  }
}

async function deleteDraft(key: string): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    await requestResult(transaction.objectStore(STORE_NAME).delete(key));
    await transactionComplete(transaction);
  } finally {
    database.close();
  }
}

export function loadPrepareWorkspace(): Promise<PrepareWorkspaceDraft | null> {
  return readDraft<PrepareWorkspaceDraft>(PREPARE_DRAFT_KEY);
}

export function savePrepareWorkspace(draft: PrepareWorkspaceDraft): Promise<void> {
  return writeDraft(PREPARE_DRAFT_KEY, draft);
}

export function clearPrepareWorkspace(): Promise<void> {
  return deleteDraft(PREPARE_DRAFT_KEY);
}

export function saveTtsHandoff(text: string): Promise<void> {
  return writeDraft(TTS_HANDOFF_KEY, { text, updatedAt: Date.now() });
}

export async function consumeTtsHandoff(): Promise<string | null> {
  const handoff = await readDraft<{ text: string; updatedAt: number }>(TTS_HANDOFF_KEY);
  if (!handoff?.text.trim()) return null;
  await deleteDraft(TTS_HANDOFF_KEY);
  return handoff.text;
}

export function loadTtsWorkspace(): Promise<TtsWorkspaceDraft | null> {
  return readDraft<TtsWorkspaceDraft>(TTS_WORKSPACE_KEY);
}

export function saveTtsWorkspace(draft: TtsWorkspaceDraft): Promise<void> {
  return writeDraft(TTS_WORKSPACE_KEY, draft);
}

export function clearTtsWorkspace(): Promise<void> {
  return deleteDraft(TTS_WORKSPACE_KEY);
}
