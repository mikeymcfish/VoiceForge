import { createHash } from "node:crypto";
import { constants as fsConstants, type Dirent, type Stats } from "node:fs";
import {
  lstat,
  open,
  readdir,
  realpath,
  type FileHandle,
} from "node:fs/promises";
import path from "node:path";

export const MAX_DEFAULT_VOICE_AUDIO_BYTES = 32 * 1024 * 1024;
export const MAX_DEFAULT_VOICE_TRANSCRIPT_BYTES = 128 * 1024;

export const SUPPORTED_DEFAULT_VOICE_FORMATS = [
  "wav",
  "mp3",
  "flac",
  "m4a",
  "aac",
  "ogg",
  "opus",
  "webm",
] as const;

export type DefaultVoiceFormat = (typeof SUPPORTED_DEFAULT_VOICE_FORMATS)[number];

/** Safe to return from an HTTP or MCP catalog response. It never contains a path. */
export interface DefaultVoiceMetadata {
  id: string;
  name: string;
  format: DefaultVoiceFormat;
  mimeType: string;
  sizeBytes: number;
  hasTranscript: boolean;
  transcript: string | null;
}

/** Backend-only resolution result for passing a catalog voice to a local worker. */
export interface ResolvedDefaultVoice {
  metadata: DefaultVoiceMetadata;
  audioPath: string;
}

/** Snapshot read that is safe to pass to upload-oriented or remote inference code. */
export interface ReadDefaultVoice {
  metadata: DefaultVoiceMetadata;
  audio: Buffer;
}

interface ValidatedFile {
  absolutePath: string;
  sizeBytes: number;
  dev: number;
  ino: number;
  mtimeMs: number;
}

interface InternalVoice {
  metadata: DefaultVoiceMetadata;
  audio: ValidatedFile;
}

const SUPPORTED_FORMAT_SET = new Set<string>(SUPPORTED_DEFAULT_VOICE_FORMATS);
const VOICE_ID_PATTERN = /^voice_[A-Za-z0-9_-]{43}$/;

const MIME_TYPES: Record<DefaultVoiceFormat, string> = {
  wav: "audio/wav",
  mp3: "audio/mpeg",
  flac: "audio/flac",
  m4a: "audio/mp4",
  aac: "audio/aac",
  ogg: "audio/ogg",
  opus: "audio/ogg",
  webm: "audio/webm",
};

function normalizeStem(stem: string): string {
  return stem.normalize("NFKC").toLocaleLowerCase("en-US");
}

function createVoiceId(normalizedStem: string): string {
  const digest = createHash("sha256")
    .update("voiceforge-default-voice:v1\0", "utf8")
    .update(normalizedStem, "utf8")
    .digest("base64url");
  return `voice_${digest}`;
}

function samePath(left: string, right: string): boolean {
  const normalizedLeft = path.normalize(left);
  const normalizedRight = path.normalize(right);
  return process.platform === "win32"
    ? normalizedLeft.toLocaleLowerCase("en-US") === normalizedRight.toLocaleLowerCase("en-US")
    : normalizedLeft === normalizedRight;
}

function isDirectChild(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative.length > 0 &&
    !path.isAbsolute(relative) &&
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    path.basename(relative) === relative
  );
}

function isSafeDirectChildName(name: string): boolean {
  return (
    name.length > 0 &&
    name !== "." &&
    name !== ".." &&
    path.basename(name) === name &&
    !name.includes("\0") &&
    !/[\u0001-\u001f\u007f]/u.test(name)
  );
}

function cloneMetadata(metadata: DefaultVoiceMetadata): DefaultVoiceMetadata {
  return { ...metadata };
}

function sameIdentity(expected: ValidatedFile, actual: Stats): boolean {
  if (!actual.isFile() || actual.size !== expected.sizeBytes) return false;

  // dev/ino are available on supported Node filesystems. If a platform reports
  // zero for both, size and mtime still detect ordinary replacement races.
  if ((expected.dev !== 0 || expected.ino !== 0) && (actual.dev !== 0 || actual.ino !== 0)) {
    return expected.dev === actual.dev && expected.ino === actual.ino;
  }
  return actual.mtimeMs === expected.mtimeMs;
}

function sameValidatedIdentity(left: ValidatedFile, right: ValidatedFile): boolean {
  return (
    left.sizeBytes === right.sizeBytes &&
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.mtimeMs === right.mtimeMs &&
    samePath(left.absolutePath, right.absolutePath)
  );
}

async function closeQuietly(handle: FileHandle): Promise<void> {
  try {
    await handle.close();
  } catch {
    // The original validation/read failure is the useful result.
  }
}

export class VoiceLibraryService {
  private readonly configuredRoot: string;

  constructor(rootDirectory = path.resolve(process.cwd(), "default_voices")) {
    this.configuredRoot = path.resolve(rootDirectory);
  }

  /** Returns a deterministic catalog containing public, path-free metadata. */
  async listVoices(): Promise<DefaultVoiceMetadata[]> {
    const voices = await this.scan();
    return voices.map(({ metadata }) => cloneMetadata(metadata));
  }

  /** Alias useful to callers that expose this collection as a catalog resource. */
  async catalog(): Promise<DefaultVoiceMetadata[]> {
    return this.listVoices();
  }

  /**
   * Resolves a catalog ID to a currently validated direct-child audio path.
   * The returned path is intentionally kept out of DefaultVoiceMetadata.
   */
  async resolveVoice(id: string): Promise<ResolvedDefaultVoice | undefined> {
    const voice = await this.findAndRevalidate(id);
    if (!voice) return undefined;
    return {
      metadata: cloneMetadata(voice.metadata),
      audioPath: voice.audio.absolutePath,
    };
  }

  /**
   * Reads a point-in-time audio snapshot after containment, symlink, identity,
   * and size revalidation. Invalid or stale IDs fail closed.
   */
  async readVoice(id: string): Promise<ReadDefaultVoice | undefined> {
    const voice = await this.findAndRevalidate(id);
    if (!voice) return undefined;

    const audio = await this.readValidatedSnapshot(
      voice.audio,
      MAX_DEFAULT_VOICE_AUDIO_BYTES
    );
    if (!audio || audio.length === 0) return undefined;

    return {
      metadata: cloneMetadata(voice.metadata),
      audio,
    };
  }

  private async findAndRevalidate(id: string): Promise<InternalVoice | undefined> {
    if (!VOICE_ID_PATTERN.test(id)) return undefined;

    const voices = await this.scan();
    const match = voices.find(({ metadata }) => metadata.id === id);
    if (!match) return undefined;

    const root = await this.validatedRoot();
    if (!root) return undefined;
    const refreshed = await this.validateDirectFile(
      root,
      path.basename(match.audio.absolutePath),
      MAX_DEFAULT_VOICE_AUDIO_BYTES,
      false
    );
    if (!refreshed || !sameValidatedIdentity(match.audio, refreshed)) return undefined;

    return { metadata: match.metadata, audio: refreshed };
  }

  private async scan(): Promise<InternalVoice[]> {
    const root = await this.validatedRoot();
    if (!root) return [];

    let entries: Dirent[];
    try {
      entries = await readdir(root, { withFileTypes: true, encoding: "utf8" });
    } catch {
      return [];
    }

    const audioGroups = new Map<string, string[]>();
    const transcriptGroups = new Map<string, string[]>();

    for (const entry of entries) {
      if (!entry.isFile() || entry.isSymbolicLink() || !isSafeDirectChildName(entry.name)) {
        continue;
      }

      const extension = path.extname(entry.name).slice(1).toLocaleLowerCase("en-US");
      const stem = path.basename(entry.name, path.extname(entry.name));
      if (!stem) continue;
      const normalizedStem = normalizeStem(stem);

      if (SUPPORTED_FORMAT_SET.has(extension)) {
        const group = audioGroups.get(normalizedStem) ?? [];
        group.push(entry.name);
        audioGroups.set(normalizedStem, group);
      } else if (extension === "txt") {
        const group = transcriptGroups.get(normalizedStem) ?? [];
        group.push(entry.name);
        transcriptGroups.set(normalizedStem, group);
      }
    }

    const voices: InternalVoice[] = [];
    const occupiedIds = new Set<string>();

    for (const normalizedStem of [...audioGroups.keys()].sort()) {
      const audioNames = audioGroups.get(normalizedStem)!;
      // Exclude every collision instead of selecting a winner based on
      // filesystem enumeration order.
      if (audioNames.length !== 1) continue;

      const audioName = audioNames[0];
      const audio = await this.validateDirectFile(
        root,
        audioName,
        MAX_DEFAULT_VOICE_AUDIO_BYTES,
        false
      );
      if (!audio) continue;

      const rawExtension = path.extname(audioName);
      const format = rawExtension.slice(1).toLocaleLowerCase("en-US") as DefaultVoiceFormat;
      const displayName = path.basename(audioName, rawExtension);
      const id = createVoiceId(normalizedStem);
      // A full SHA-256 collision is extraordinarily unlikely, but excluding
      // either result remains safer than exposing an ambiguous identifier.
      if (occupiedIds.has(id)) continue;

      let transcript: string | null = null;
      const transcriptNames = transcriptGroups.get(normalizedStem) ?? [];
      if (transcriptNames.length === 1) {
        transcript = await this.readTranscript(root, transcriptNames[0]);
      }

      occupiedIds.add(id);
      voices.push({
        metadata: {
          id,
          name: displayName,
          format,
          mimeType: MIME_TYPES[format],
          sizeBytes: audio.sizeBytes,
          hasTranscript: transcript !== null,
          transcript,
        },
        audio,
      });
    }

    return voices;
  }

  private async validatedRoot(): Promise<string | undefined> {
    try {
      const rootStat = await lstat(this.configuredRoot);
      if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) return undefined;
      const resolved = await realpath(this.configuredRoot);
      if (!samePath(resolved, this.configuredRoot)) return undefined;
      return resolved;
    } catch {
      return undefined;
    }
  }

  private async validateDirectFile(
    root: string,
    fileName: string,
    maximumBytes: number,
    allowEmpty: boolean
  ): Promise<ValidatedFile | undefined> {
    if (!isSafeDirectChildName(fileName)) return undefined;
    const candidate = path.resolve(root, fileName);
    if (!isDirectChild(root, candidate)) return undefined;

    try {
      const fileStat = await lstat(candidate);
      if (!fileStat.isFile() || fileStat.isSymbolicLink()) return undefined;
      if ((!allowEmpty && fileStat.size === 0) || fileStat.size > maximumBytes) return undefined;

      const resolved = await realpath(candidate);
      if (!samePath(resolved, candidate) || !isDirectChild(root, resolved)) return undefined;

      return {
        absolutePath: candidate,
        sizeBytes: fileStat.size,
        dev: fileStat.dev,
        ino: fileStat.ino,
        mtimeMs: fileStat.mtimeMs,
      };
    } catch {
      return undefined;
    }
  }

  private async readTranscript(root: string, fileName: string): Promise<string | null> {
    const transcriptFile = await this.validateDirectFile(
      root,
      fileName,
      MAX_DEFAULT_VOICE_TRANSCRIPT_BYTES,
      true
    );
    if (!transcriptFile) return null;

    const contents = await this.readValidatedSnapshot(
      transcriptFile,
      MAX_DEFAULT_VOICE_TRANSCRIPT_BYTES
    );
    if (!contents) return null;

    try {
      const text = new TextDecoder("utf-8", { fatal: true }).decode(contents);
      if (text.includes("\0")) return null;
      return text.replace(/^\uFEFF/u, "");
    } catch {
      return null;
    }
  }

  private async readValidatedSnapshot(
    expected: ValidatedFile,
    maximumBytes: number
  ): Promise<Buffer | undefined> {
    let handle: FileHandle | undefined;
    try {
      handle = await open(expected.absolutePath, fsConstants.O_RDONLY);
      const before = await handle.stat();
      if (!sameIdentity(expected, before) || before.size > maximumBytes) return undefined;

      const contents = await handle.readFile();
      const after = await handle.stat();
      if (
        !sameIdentity(expected, after) ||
        contents.length !== before.size ||
        contents.length > maximumBytes
      ) {
        return undefined;
      }
      return contents;
    } catch {
      return undefined;
    } finally {
      if (handle) await closeQuietly(handle);
    }
  }
}

export const voiceLibraryService = new VoiceLibraryService(
  process.env.VOICEFORGE_DEFAULT_VOICES_DIR || undefined
);
