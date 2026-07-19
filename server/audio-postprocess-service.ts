import fs from "node:fs";
import fsPromises from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { terminateChildProcess } from "./process-utils";

export type ReferenceEnhancementMode = "none" | "cleanup" | "audiosr";
export type SpeechOutputFormat = "wav" | "mp3";
export type AudioSrModel = "speech" | "basic";
export type AudioSrDevice = "auto" | "cpu" | "cuda" | "mps" | `cuda:${number}`;

export interface WorkerChapter {
  title: string | null;
  start_sample: number;
  start_ms: number;
  start_seconds: number;
}

export interface WorkerChapterManifest {
  version: 1;
  sample_rate: number;
  total_samples: number;
  chapters: WorkerChapter[];
}

export interface AudioSrOptions {
  model?: AudioSrModel;
  device?: AudioSrDevice;
  steps?: number;
  guidance?: number;
  seed?: number;
}

export interface AudioProcessHooks {
  onSpawn?: (child: ChildProcess) => void;
  assertActive?: () => void;
  onLog?: (message: string) => void;
}

export interface EnhanceReferenceAudioOptions extends AudioProcessHooks {
  inputPath: string;
  workingDir: string;
  mode: ReferenceEnhancementMode;
  /** Preferred grouped form for direct service callers. */
  audioSr?: AudioSrOptions;
  /** Flat aliases used by the speech request pipeline. */
  audioSrModel?: AudioSrModel;
  audioSrDevice?: string;
  audioSrDdimSteps?: number;
  audioSrGuidanceScale?: number;
  audioSrSeed?: number;
}

export interface FinalizeSpeechAudioOptions extends AudioProcessHooks {
  inputWavPath: string;
  workingDir: string;
  outputFormat: SpeechOutputFormat;
  normalizeLevels?: boolean;
  chapterManifestPath?: string;
  mp3Quality?: number;
}

export interface FinalizedSpeechAudio {
  outputPath: string;
  format: SpeechOutputFormat;
  mimeType: "audio/wav" | "audio/mpeg";
  chapterCount: number;
}

export interface AudioProcessingCapabilities {
  ffmpegAvailable: boolean;
  audioSrAvailable: boolean;
}

export interface BuiltFfmetadata {
  text: string;
  chapterCount: number;
}

type BinaryResolution = {
  executable: string;
  source: string;
};

type BinaryResolutionResult =
  | { resolution: BinaryResolution; error?: never }
  | { resolution?: never; error: string };

const MAX_MANIFEST_BYTES = 1024 * 1024;
const MAX_CHAPTERS = 10_000;
const MAX_CHAPTER_TITLE_LENGTH = 4_096;
const MAX_SAMPLE_RATE = 768_000;
const MAX_PROCESS_LOG_LINES = 200;
const MAX_PROCESS_LOG_CHARS = 64 * 1024;
const MAX_PROCESS_LOG_LINE_CHARS = 2_048;
const MAX_AUDIO_SR_SCAN_ENTRIES = 2_000;
const MAX_AUDIO_SR_SCAN_DEPTH = 6;
const MAX_MANAGED_ARTIFACT_BYTES = 250 * 1024 * 1024;
const MAX_WAV_HEADER_BYTES = 1024 * 1024;
const AUDIO_SR_SUFFIX = "_voiceforge_audiosr";

export const CONSERVATIVE_CLEANUP_FILTER = [
  "highpass=f=70",
  "afftdn=nr=8:nf=-35:tn=1",
  [
    "silenceremove=start_periods=1",
    "start_threshold=-40dB",
  ].join(":"),
  // A positive stop_periods value stops at the first qualifying pause; it
  // does not distinguish an interior pause from the trailing edge. Reverse
  // the clip and apply another leading-edge trim so interior pauses survive.
  "areverse",
  [
    "silenceremove=start_periods=1",
    "start_threshold=-42dB",
  ].join(":"),
  "areverse",
  "loudnorm=I=-20:LRA=7:TP=-1.5:linear=true",
  "alimiter=limit=0.95:attack=5:release=50:level=true",
].join(",");

export const SPEECH_LEVEL_NORMALIZATION_FILTER =
  "loudnorm=I=-18:LRA=7:TP=-1.5";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string
): void {
  const expectedSet = new Set(expected);
  const actual = Object.keys(value);
  const unknown = actual.filter((key) => !expectedSet.has(key));
  const missing = expected.filter((key) => !Object.prototype.hasOwnProperty.call(value, key));
  if (unknown.length > 0 || missing.length > 0) {
    const details = [
      unknown.length > 0 ? `unknown fields: ${unknown.join(", ")}` : "",
      missing.length > 0 ? `missing fields: ${missing.join(", ")}` : "",
    ]
      .filter(Boolean)
      .join("; ");
    throw new Error(`${label} has an unsupported structure (${details}).`);
  }
}

function requireSafeInteger(
  value: unknown,
  label: string,
  minimum: number,
  maximum = Number.MAX_SAFE_INTEGER
): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new Error(`${label} must be an integer between ${minimum} and ${maximum}.`);
  }
  return value as number;
}

function requireFiniteNumber(
  value: unknown,
  label: string,
  minimum: number,
  maximum = Number.MAX_VALUE
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new Error(`${label} must be a finite number between ${minimum} and ${maximum}.`);
  }
  return value;
}

/**
 * Strictly validates the versioned manifest written by the Qwen/MOSS workers.
 * Sample offsets are canonical; the millisecond and second fields must agree.
 */
export function validateWorkerChapterManifest(value: unknown): WorkerChapterManifest {
  if (!isRecord(value)) throw new Error("Chapter manifest must be a JSON object.");
  assertExactKeys(value, ["version", "sample_rate", "total_samples", "chapters"], "Chapter manifest");
  if (value.version !== 1) throw new Error("Chapter manifest version must be exactly 1.");

  const sampleRate = requireSafeInteger(value.sample_rate, "sample_rate", 1, MAX_SAMPLE_RATE);
  const totalSamples = requireSafeInteger(value.total_samples, "total_samples", 1);
  if (!Array.isArray(value.chapters)) throw new Error("chapters must be an array.");
  if (value.chapters.length > MAX_CHAPTERS) {
    throw new Error(`chapters contains more than ${MAX_CHAPTERS} entries.`);
  }

  let previousStartSample = -1;
  const chapters = value.chapters.map((rawChapter, index): WorkerChapter => {
    const label = `chapters[${index}]`;
    if (!isRecord(rawChapter)) throw new Error(`${label} must be an object.`);
    assertExactKeys(
      rawChapter,
      ["title", "start_sample", "start_ms", "start_seconds"],
      label
    );
    if (rawChapter.title !== null && typeof rawChapter.title !== "string") {
      throw new Error(`${label}.title must be null or a string.`);
    }
    if (typeof rawChapter.title === "string") {
      if (!rawChapter.title.trim()) {
        throw new Error(`${label}.title must be null or a non-empty string.`);
      }
      if (rawChapter.title.length > MAX_CHAPTER_TITLE_LENGTH) {
        throw new Error(`${label}.title exceeds ${MAX_CHAPTER_TITLE_LENGTH} characters.`);
      }
      if (/[\u0000\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/u.test(rawChapter.title)) {
        throw new Error(`${label}.title contains unsupported control characters.`);
      }
    }

    const startSample = requireSafeInteger(rawChapter.start_sample, `${label}.start_sample`, 0);
    const startMs = requireFiniteNumber(rawChapter.start_ms, `${label}.start_ms`, 0);
    const startSeconds = requireFiniteNumber(
      rawChapter.start_seconds,
      `${label}.start_seconds`,
      0
    );
    if (startSample >= totalSamples) {
      throw new Error(`${label}.start_sample must be less than total_samples.`);
    }
    if (startSample <= previousStartSample) {
      throw new Error("Chapter start_sample values must be strictly increasing.");
    }

    const expectedSeconds = startSample / sampleRate;
    const expectedMs = (startSample * 1000) / sampleRate;
    if (Math.abs(startSeconds - expectedSeconds) > 1e-12) {
      throw new Error(`${label}.start_seconds does not match start_sample/sample_rate.`);
    }
    if (Math.abs(startMs - expectedMs) > 1e-9) {
      throw new Error(`${label}.start_ms does not match its sample-derived timestamp.`);
    }

    previousStartSample = startSample;
    return {
      title: rawChapter.title,
      start_sample: startSample,
      start_ms: startMs,
      start_seconds: startSeconds,
    };
  });

  return {
    version: 1,
    sample_rate: sampleRate,
    total_samples: totalSamples,
    chapters,
  };
}

export function parseWorkerChapterManifest(json: string): WorkerChapterManifest {
  if (Buffer.byteLength(json, "utf8") > MAX_MANIFEST_BYTES) {
    throw new Error(`Chapter manifest exceeds ${MAX_MANIFEST_BYTES} bytes.`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Chapter manifest is not valid JSON: ${detail}`);
  }
  return validateWorkerChapterManifest(parsed);
}

function sanitizeChapterTitle(value: string | null): string | undefined {
  if (value === null) return undefined;
  const normalized = value.trim().replace(/[ \t]+/g, " ");
  return normalized || undefined;
}

/** Escapes values using FFmpeg's ffmetadata rules, matching the upstream implementation. */
export function escapeFfmetadataValue(value: string): string {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/\n/g, " ")
    .replace(/\r/g, " ")
    .replace(/;/g, "\\;")
    .replace(/#/g, "\\#")
    .replace(/=/g, "\\=");
}

/**
 * Builds the exact upstream chapter document: millisecond timebase, sorted and
 * de-duplicated starts, next-start end times, and generated fallback titles.
 */
export function buildFfmetadataDocument(manifestValue: WorkerChapterManifest): BuiltFfmetadata {
  const manifest = validateWorkerChapterManifest(manifestValue);
  const totalMs = Math.max(
    0,
    Math.round((manifest.total_samples / manifest.sample_rate) * 1000)
  );
  const normalized = manifest.chapters
    .map((chapter) => ({
      startMs: Math.max(0, Math.round((chapter.start_sample / manifest.sample_rate) * 1000)),
      title: sanitizeChapterTitle(chapter.title),
    }))
    .filter((chapter) => chapter.startMs < totalMs)
    .sort((left, right) => left.startMs - right.startMs);

  const deduped: typeof normalized = [];
  const seenStarts = new Set<number>();
  for (const chapter of normalized) {
    if (seenStarts.has(chapter.startMs)) continue;
    seenStarts.add(chapter.startMs);
    deduped.push(chapter);
  }

  const lines = [";FFMETADATA1"];
  deduped.forEach((chapter, zeroBasedIndex) => {
    const oneBasedIndex = zeroBasedIndex + 1;
    const nextStartMs =
      zeroBasedIndex + 1 < deduped.length
        ? deduped[zeroBasedIndex + 1]!.startMs
        : Math.max(chapter.startMs + 1, totalMs);
    const endMs = Math.max(chapter.startMs + 1, nextStartMs);
    const title = chapter.title || `Chapter ${oneBasedIndex}`;
    lines.push(
      "[CHAPTER]",
      "TIMEBASE=1/1000",
      `START=${chapter.startMs}`,
      `END=${endMs}`,
      `title=${escapeFfmetadataValue(title)}`
    );
  });

  return {
    text: `${lines.join("\n")}\n`,
    chapterCount: deduped.length,
  };
}

export function buildFfmetadata(manifest: WorkerChapterManifest): string {
  return buildFfmetadataDocument(manifest).text;
}

function requireNonEmptyPath(value: string, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} is required.`);
  if (value.includes("\0")) throw new Error(`${label} contains an invalid NUL character.`);
  return value;
}

function normalizeMp3Quality(value: number | undefined): number {
  const quality = value ?? 0;
  if (!Number.isInteger(quality) || quality < 0 || quality > 9) {
    throw new Error("mp3Quality must be an integer from 0 (best) through 9 (smallest).");
  }
  return quality;
}

export function buildMp3FfmpegArgs(options: {
  inputWavPath: string;
  outputMp3Path: string;
  ffmetadataPath?: string;
  mp3Quality?: number;
  normalizeLevels?: boolean;
  sampleRate?: number;
}): string[] {
  const input = requireNonEmptyPath(options.inputWavPath, "inputWavPath");
  const output = requireNonEmptyPath(options.outputMp3Path, "outputMp3Path");
  const quality = normalizeMp3Quality(options.mp3Quality);
  const args = ["-y", "-hide_banner", "-loglevel", "error", "-i", input];
  if (options.ffmetadataPath) {
    args.push(
      "-f",
      "ffmetadata",
      "-i",
      requireNonEmptyPath(options.ffmetadataPath, "ffmetadataPath"),
      "-map_metadata",
      "1",
      "-map_chapters",
      "1",
      "-id3v2_version",
      "3"
    );
  }
  args.push("-map", "0:a", "-vn");
  if (options.normalizeLevels) {
    const sampleRate = requireSafeInteger(
      options.sampleRate,
      "sampleRate",
      1,
      MAX_SAMPLE_RATE
    );
    args.push(
      "-af",
      SPEECH_LEVEL_NORMALIZATION_FILTER,
      "-ar",
      String(sampleRate)
    );
  }
  args.push("-codec:a", "libmp3lame");
  if (options.ffmetadataPath) {
    args.push("-b:a", "192k", "-write_xing", "0");
  } else {
    args.push("-q:a", String(quality));
  }
  args.push(output);
  return args;
}

export function buildLevelNormalizedWavFfmpegArgs(options: {
  inputWavPath: string;
  outputWavPath: string;
  sampleRate: number;
}): string[] {
  const sampleRate = requireSafeInteger(
    options.sampleRate,
    "sampleRate",
    1,
    MAX_SAMPLE_RATE
  );
  return [
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    requireNonEmptyPath(options.inputWavPath, "inputWavPath"),
    "-map",
    "0:a",
    "-vn",
    "-af",
    SPEECH_LEVEL_NORMALIZATION_FILTER,
    "-ar",
    String(sampleRate),
    "-map_metadata",
    "-1",
    "-codec:a",
    "pcm_s16le",
    requireNonEmptyPath(options.outputWavPath, "outputWavPath"),
  ];
}

export function buildCleanupFfmpegArgs(options: {
  inputPath: string;
  outputWavPath: string;
}): string[] {
  return [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    requireNonEmptyPath(options.inputPath, "inputPath"),
    "-vn",
    "-ac",
    "1",
    "-ar",
    "24000",
    "-sample_fmt",
    "s16",
    "-af",
    CONSERVATIVE_CLEANUP_FILTER,
    "-map_metadata",
    "-1",
    "-codec:a",
    "pcm_s16le",
    requireNonEmptyPath(options.outputWavPath, "outputWavPath"),
  ];
}

function normalizeAudioSrOptions(options: AudioSrOptions | undefined): Required<AudioSrOptions> {
  const model = options?.model ?? "speech";
  if (model !== "speech" && model !== "basic") {
    throw new Error("AudioSR model must be either speech or basic.");
  }

  const rawDevice = options?.device ?? "auto";
  const device = rawDevice.toLowerCase();
  if (!/^(?:auto|cpu|cuda|mps|cuda:\d{1,3})$/u.test(device)) {
    throw new Error("AudioSR device must be auto, cpu, cuda, mps, or cuda:<index>.");
  }

  const steps = options?.steps ?? 50;
  if (!Number.isInteger(steps) || steps < 10 || steps > 250) {
    throw new Error("AudioSR steps must be an integer from 10 through 250.");
  }

  const guidance = options?.guidance ?? 3.5;
  if (!Number.isFinite(guidance) || guidance < 1 || guidance > 10) {
    throw new Error("AudioSR guidance must be between 1 and 10.");
  }

  const seed = options?.seed ?? 42;
  if (!Number.isSafeInteger(seed) || seed < -2_147_483_648 || seed > 2_147_483_647) {
    throw new Error("AudioSR seed must be a signed 32-bit integer.");
  }

  return { model, device: device as AudioSrDevice, steps, guidance, seed };
}

export function buildAudioSrArgs(options: {
  inputPath: string;
  outputDirectory: string;
  audioSr?: AudioSrOptions;
}): string[] {
  const audioSr = normalizeAudioSrOptions(options.audioSr);
  return [
    "-i",
    requireNonEmptyPath(options.inputPath, "inputPath"),
    "-s",
    requireNonEmptyPath(options.outputDirectory, "outputDirectory"),
    "--model_name",
    audioSr.model,
    "-d",
    audioSr.device,
    "--ddim_steps",
    String(audioSr.steps),
    "--guidance_scale",
    String(audioSr.guidance),
    "--seed",
    String(audioSr.seed),
    "--suffix",
    AUDIO_SR_SUFFIX,
  ];
}

function envValue(env: NodeJS.ProcessEnv, key: string): string | undefined {
  const direct = env[key];
  if (direct !== undefined) return direct;
  const match = Object.keys(env).find((candidate) => candidate.toUpperCase() === key);
  return match ? env[match] : undefined;
}

function unquoteConfiguredPath(value: string): string {
  const trimmed = value.trim();
  if (
    trimmed.length >= 2 &&
    ((trimmed.startsWith('"') && trimmed.endsWith('"')) ||
      (trimmed.startsWith("'") && trimmed.endsWith("'")))
  ) {
    return trimmed.slice(1, -1).trim();
  }
  return trimmed;
}

function isExecutableFile(candidate: string): boolean {
  try {
    const stats = fs.statSync(candidate);
    if (!stats.isFile()) return false;
    if (process.platform !== "win32") fs.accessSync(candidate, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function nativeExecutableNames(baseName: string): string[] {
  return process.platform === "win32" ? [`${baseName}.exe`, baseName] : [baseName];
}

function resolveExistingCandidate(
  configuredValue: string,
  baseName: string,
  env: NodeJS.ProcessEnv
): string | undefined {
  const configured = unquoteConfiguredPath(configuredValue);
  if (!configured) return undefined;

  try {
    const stats = fs.statSync(configured);
    if (stats.isFile() && isExecutableFile(configured)) return path.resolve(configured);
    if (stats.isDirectory()) {
      for (const name of nativeExecutableNames(baseName)) {
        const nested = path.join(configured, name);
        if (isExecutableFile(nested)) return path.resolve(nested);
      }
      return undefined;
    }
  } catch {
    // A simple command name can still be resolved through PATH.
  }

  if (!configured.includes("/") && !configured.includes("\\")) {
    return searchPath(configured, env);
  }
  return undefined;
}

function searchPath(baseName: string, env: NodeJS.ProcessEnv): string | undefined {
  const rawPath = envValue(env, "PATH") || "";
  const requestedHasExtension = path.extname(baseName).length > 0;
  const names = requestedHasExtension ? [baseName] : nativeExecutableNames(baseName);
  for (const directory of rawPath.split(path.delimiter)) {
    const cleanDirectory = unquoteConfiguredPath(directory);
    if (!cleanDirectory) continue;
    for (const name of names) {
      const candidate = path.join(cleanDirectory, name);
      if (isExecutableFile(candidate)) return path.resolve(candidate);
    }
  }
  return undefined;
}

function resolveFfmpegResult(env: NodeJS.ProcessEnv = process.env): BinaryResolutionResult {
  for (const envName of ["VOICEFORGE_FFMPEG_BIN", "MOSS_TTS_FFMPEG_BIN"] as const) {
    const configured = envValue(env, envName)?.trim();
    if (!configured) continue;
    const executable = resolveExistingCandidate(configured, "ffmpeg", env);
    if (executable) return { resolution: { executable, source: envName } };
    return { error: `${envName} does not resolve to a native ffmpeg executable.` };
  }
  const executable = searchPath("ffmpeg", env);
  return executable
    ? { resolution: { executable, source: "PATH" } }
    : { error: "ffmpeg was not found in PATH." };
}

function configuredPythonAudioSrCandidates(
  env: NodeJS.ProcessEnv,
  pythonEnvName: "QWEN_TTS_PYTHON" | "MOSS_TTS_PYTHON"
): string[] {
  const configured = envValue(env, pythonEnvName)?.trim();
  if (!configured) return [];
  const pythonPath = unquoteConfiguredPath(configured);
  if (!path.isAbsolute(pythonPath)) return [];
  const pythonDirectory = path.dirname(pythonPath);
  const parentDirectory = path.dirname(pythonDirectory);
  const directories = [
    pythonDirectory,
    path.join(pythonDirectory, "Scripts"),
    path.join(parentDirectory, "Scripts"),
  ];
  return directories.flatMap((directory) =>
    nativeExecutableNames("audiosr").map((name) => path.join(directory, name))
  );
}

function resolveAudioSrResult(env: NodeJS.ProcessEnv = process.env): BinaryResolutionResult {
  const configured = envValue(env, "VOICEFORGE_AUDIOSR_BIN")?.trim();
  if (configured) {
    const executable = resolveExistingCandidate(configured, "audiosr", env);
    return executable
      ? { resolution: { executable, source: "VOICEFORGE_AUDIOSR_BIN" } }
      : {
          error:
            "VOICEFORGE_AUDIOSR_BIN does not resolve to a native AudioSR executable.",
        };
  }

  const pathExecutable = searchPath("audiosr", env);
  if (pathExecutable) return { resolution: { executable: pathExecutable, source: "PATH" } };

  for (const pythonEnvName of ["QWEN_TTS_PYTHON", "MOSS_TTS_PYTHON"] as const) {
    for (const candidate of configuredPythonAudioSrCandidates(env, pythonEnvName)) {
      if (isExecutableFile(candidate)) {
        return {
          resolution: { executable: path.resolve(candidate), source: `${pythonEnvName}:Scripts` },
        };
      }
    }
  }
  return {
    error:
      "AudioSR was not found in VOICEFORGE_AUDIOSR_BIN, PATH, or a configured speech runtime.",
  };
}

export function resolveFfmpegBinary(env: NodeJS.ProcessEnv = process.env): string {
  const result = resolveFfmpegResult(env);
  if (!result.resolution) throw new Error(result.error);
  return result.resolution.executable;
}

export function resolveAudioSrBinary(env: NodeJS.ProcessEnv = process.env): string {
  const result = resolveAudioSrResult(env);
  if (!result.resolution) throw new Error(result.error);
  return result.resolution.executable;
}

export function getAudioProcessingCapabilities(
  env: NodeJS.ProcessEnv = process.env
): AudioProcessingCapabilities {
  return {
    ffmpegAvailable: Boolean(resolveFfmpegResult(env).resolution),
    audioSrAvailable: Boolean(resolveAudioSrResult(env).resolution),
  };
}

function samePath(left: string, right: string): boolean {
  const normalizedLeft = path.normalize(left);
  const normalizedRight = path.normalize(right);
  return process.platform === "win32"
    ? normalizedLeft.toLocaleLowerCase("en-US") ===
        normalizedRight.toLocaleLowerCase("en-US")
    : normalizedLeft === normalizedRight;
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative.length > 0 &&
    !path.isAbsolute(relative) &&
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`)
  );
}

async function resolveManagedWorkingDirectory(workingDir: string): Promise<string> {
  const absolute = path.resolve(requireNonEmptyPath(workingDir, "workingDir"));
  const stats = await fsPromises.lstat(absolute).catch(() => undefined);
  if (!stats?.isDirectory() || stats.isSymbolicLink()) {
    throw new Error("workingDir must be an existing, non-symlink directory.");
  }
  const resolved = await fsPromises.realpath(absolute);
  if (!samePath(absolute, resolved)) {
    throw new Error("workingDir must not traverse symbolic links.");
  }
  return resolved;
}

async function resolveManagedFile(
  root: string,
  candidatePath: string,
  label: string,
  maximumBytes?: number
): Promise<string> {
  const rawCandidate = requireNonEmptyPath(candidatePath, label);
  const candidate = path.resolve(
    path.isAbsolute(rawCandidate) ? rawCandidate : path.join(root, rawCandidate)
  );
  if (!isWithin(root, candidate)) {
    throw new Error(`${label} must be contained by workingDir.`);
  }
  const stats = await fsPromises.lstat(candidate).catch(() => undefined);
  if (!stats?.isFile() || stats.isSymbolicLink()) {
    throw new Error(`${label} must be an existing, non-symlink file.`);
  }
  if (maximumBytes !== undefined && stats.size > maximumBytes) {
    throw new Error(`${label} exceeds ${maximumBytes} bytes.`);
  }
  const resolved = await fsPromises.realpath(candidate);
  if (!samePath(candidate, resolved) || !isWithin(root, resolved)) {
    throw new Error(`${label} must not traverse symbolic links or leave workingDir.`);
  }
  return resolved;
}

function managedOutputPath(root: string, fileName: string): string {
  if (path.basename(fileName) !== fileName || fileName.includes("\0")) {
    throw new Error("Managed output filename is invalid.");
  }
  const candidate = path.join(root, fileName);
  if (!isWithin(root, candidate)) throw new Error("Managed output path escaped workingDir.");
  return candidate;
}

function temporaryOutputPath(root: string, stem: string, extension: string): string {
  return managedOutputPath(root, `.${stem}.${randomUUID()}.tmp${extension}`);
}

async function validateProducedFile(root: string, candidate: string, label: string): Promise<void> {
  const resolved = await resolveManagedFile(root, candidate, label);
  const stats = await fsPromises.stat(resolved);
  if (stats.size <= 0) throw new Error(`${label} is empty.`);
  if (stats.size > MAX_MANAGED_ARTIFACT_BYTES) {
    throw new Error(`${label} exceeds the 250 MB managed-artifact limit.`);
  }
}

async function atomicReplaceManagedFile(
  root: string,
  temporaryPath: string,
  finalPath: string
): Promise<void> {
  if (!isWithin(root, temporaryPath) || !isWithin(root, finalPath)) {
    throw new Error("Atomic output paths must remain inside workingDir.");
  }
  await validateProducedFile(root, temporaryPath, "Temporary output");

  const existing = await fsPromises.lstat(finalPath).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  if (existing && (!existing.isFile() || existing.isSymbolicLink())) {
    throw new Error("Refusing to replace a non-file or symlink managed output.");
  }

  try {
    await fsPromises.rename(temporaryPath, finalPath);
    return;
  } catch (error) {
    if (!existing) throw error;
  }

  // Windows can refuse rename-over-existing. Preserve the previous complete
  // file until the new complete file is ready to take its place.
  const backupPath = managedOutputPath(
    root,
    `.${path.basename(finalPath)}.${randomUUID()}.backup`
  );
  let backupCreated = false;
  try {
    await fsPromises.rename(finalPath, backupPath);
    backupCreated = true;
    await fsPromises.rename(temporaryPath, finalPath);
    await fsPromises.rm(backupPath, { force: true }).catch(() => undefined);
  } catch (error) {
    if (backupCreated) {
      await fsPromises.rename(backupPath, finalPath).catch(() => undefined);
    }
    throw error;
  }
}

function processEnvironment(): NodeJS.ProcessEnv {
  const exactKeys = new Set([
    "ALLUSERSPROFILE",
    "ALL_PROXY",
    "APPDATA",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)",
    "COMSPEC",
    "CURL_CA_BUNDLE",
    "HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LOCALAPPDATA",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
  ]);
  const allowedPrefixes = [
    "CUDA_",
    "HSA_",
    "KMP_",
    "MKL_",
    "NVIDIA_",
    "OMP_",
    "PYTORCH_",
    "ROCR_",
    "TORCH_",
  ];
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    const normalized = key.toUpperCase();
    if (exactKeys.has(normalized) || allowedPrefixes.some((prefix) => normalized.startsWith(prefix))) {
      env[key] = value;
    }
  }
  env.PYTHONIOENCODING = "utf-8";
  env.PYTHONUTF8 = "1";
  env.PYTHONUNBUFFERED = "1";
  return env;
}

class BoundedLogSink {
  private emittedLines = 0;
  private emittedChars = 0;
  private truncationReported = false;
  private readonly tail: string[] = [];

  constructor(private readonly onLog?: (message: string) => void) {}

  public append(source: "stdout" | "stderr", rawLine: string): void {
    const cleaned = rawLine.replace(/\u0000/g, "").trim();
    if (!cleaned) return;
    const clipped =
      cleaned.length > MAX_PROCESS_LOG_LINE_CHARS
        ? `${cleaned.slice(0, MAX_PROCESS_LOG_LINE_CHARS)}…`
        : cleaned;
    const line = source === "stderr" ? `[stderr] ${clipped}` : clipped;
    this.tail.push(line);
    if (this.tail.length > 40) this.tail.shift();

    if (
      this.emittedLines >= MAX_PROCESS_LOG_LINES ||
      this.emittedChars + line.length > MAX_PROCESS_LOG_CHARS
    ) {
      if (!this.truncationReported) {
        this.truncationReported = true;
        this.onLog?.("[process output truncated]");
      }
      return;
    }
    this.emittedLines += 1;
    this.emittedChars += line.length;
    this.onLog?.(line);
  }

  public detail(): string {
    return this.tail.join("\n").slice(-16_384);
  }
}

function attachBoundedStream(
  stream: NodeJS.ReadableStream | null,
  source: "stdout" | "stderr",
  sink: BoundedLogSink,
  onCallbackError: (error: Error) => void
): () => void {
  let buffer = "";
  if (!stream) return () => undefined;
  stream.setEncoding("utf8");
  stream.on("data", (chunk: string) => {
    buffer += chunk;
    let newlineIndex = buffer.search(/\r?\n/u);
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex);
      const newlineLength = buffer[newlineIndex] === "\r" && buffer[newlineIndex + 1] === "\n" ? 2 : 1;
      buffer = buffer.slice(newlineIndex + newlineLength);
      try {
        sink.append(source, line);
      } catch (error) {
        onCallbackError(error instanceof Error ? error : new Error(String(error)));
      }
      newlineIndex = buffer.search(/\r?\n/u);
    }
    if (buffer.length > MAX_PROCESS_LOG_LINE_CHARS * 2) {
      const oversized = buffer;
      buffer = "";
      try {
        sink.append(source, oversized);
      } catch (error) {
        onCallbackError(error instanceof Error ? error : new Error(String(error)));
      }
    }
  });
  return () => {
    if (!buffer) return;
    try {
      sink.append(source, buffer);
    } catch (error) {
      onCallbackError(error instanceof Error ? error : new Error(String(error)));
    }
    buffer = "";
  };
}

async function runControlledProcess(options: {
  executable: string;
  args: string[];
  cwd: string;
  label: string;
  hooks: AudioProcessHooks;
}): Promise<void> {
  options.hooks.assertActive?.();
  await new Promise<void>((resolve, reject) => {
    let child: ChildProcess;
    try {
      child = spawn(options.executable, options.args, {
        cwd: options.cwd,
        detached: process.platform !== "win32",
        env: processEnvironment(),
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (error) {
      reject(error);
      return;
    }

    const sink = new BoundedLogSink(options.hooks.onLog);
    let callbackError: Error | undefined;
    let settled = false;
    const stopForCallbackError = (error: Error) => {
      if (callbackError) return;
      callbackError = error;
      terminateChildProcess(child, { processGroup: process.platform !== "win32" });
    };
    const flushStdout = attachBoundedStream(
      child.stdout,
      "stdout",
      sink,
      stopForCallbackError
    );
    const flushStderr = attachBoundedStream(
      child.stderr,
      "stderr",
      sink,
      stopForCallbackError
    );
    const activityTimer = setInterval(() => {
      try {
        options.hooks.assertActive?.();
      } catch (error) {
        stopForCallbackError(error instanceof Error ? error : new Error(String(error)));
      }
    }, 250);
    activityTimer.unref?.();

    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearInterval(activityTimer);
      flushStdout();
      flushStderr();
      if (error) reject(error);
      else resolve();
    };

    child.once("error", (error) => finish(error));
    child.once("close", (code, signal) => {
      if (callbackError) {
        finish(callbackError);
        return;
      }
      if (code === 0) {
        finish();
        return;
      }
      const detail = sink.detail();
      const status = code === null ? `signal ${signal || "unknown"}` : `code ${code}`;
      finish(
        new Error(
          `${options.label} exited with ${status}.${detail ? `\n${detail}` : ""}`
        )
      );
    });

    try {
      options.hooks.onSpawn?.(child);
    } catch (error) {
      stopForCallbackError(error instanceof Error ? error : new Error(String(error)));
    }
  });
  options.hooks.assertActive?.();
}

async function loadManagedManifest(
  root: string,
  manifestPath: string
): Promise<WorkerChapterManifest> {
  const resolved = await resolveManagedFile(
    root,
    manifestPath,
    "chapterManifestPath",
    MAX_MANIFEST_BYTES
  );
  const contents = await fsPromises.readFile(resolved, "utf8");
  return parseWorkerChapterManifest(contents);
}

async function findAudioSrOutput(root: string, notBeforeMs: number): Promise<string> {
  const wavFiles: string[] = [];
  let entriesVisited = 0;

  const visit = async (directory: string, depth: number): Promise<void> => {
    if (depth > MAX_AUDIO_SR_SCAN_DEPTH) {
      throw new Error("AudioSR output directory nesting is unexpectedly deep.");
    }
    const entries = await fsPromises.readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      entriesVisited += 1;
      if (entriesVisited > MAX_AUDIO_SR_SCAN_ENTRIES) {
        throw new Error("AudioSR produced too many output directory entries.");
      }
      const candidate = path.join(directory, entry.name);
      if (!isWithin(root, candidate)) throw new Error("AudioSR output escaped its managed directory.");
      if (entry.isSymbolicLink()) {
        throw new Error("AudioSR output must not contain symbolic links.");
      }
      if (entry.isDirectory()) {
        await visit(candidate, depth + 1);
      } else if (entry.isFile() && path.extname(entry.name).toLowerCase() === ".wav") {
        await resolveManagedFile(root, candidate, "AudioSR output");
        const stats = await fsPromises.stat(candidate);
        // The output directory is unique and asserted empty before launch. The
        // timestamp check is an additional guard against externally planted or
        // timestamp-stale files being selected as this run's output.
        if (stats.mtimeMs + 2_000 >= notBeforeMs) wavFiles.push(candidate);
      }
    }
  };

  await visit(root, 0);
  const suffixed = wavFiles.filter((candidate) =>
    path.basename(candidate, path.extname(candidate)).includes(AUDIO_SR_SUFFIX)
  );
  const candidates = suffixed.length > 0 ? suffixed : wavFiles;
  if (candidates.length === 0) throw new Error("AudioSR completed without producing a WAV file.");
  if (candidates.length !== 1) {
    throw new Error(`AudioSR produced ${candidates.length} candidate WAV files; expected exactly one.`);
  }
  return candidates[0]!;
}

export async function enhanceReferenceAudio(
  options: EnhanceReferenceAudioOptions
): Promise<string> {
  const root = await resolveManagedWorkingDirectory(options.workingDir);
  const input = await resolveManagedFile(root, options.inputPath, "inputPath");
  options.assertActive?.();

  if (options.mode === "none") return input;
  if (options.mode !== "cleanup" && options.mode !== "audiosr") {
    throw new Error("Reference enhancement mode must be none, cleanup, or audiosr.");
  }

  if (options.mode === "cleanup") {
    const ffmpeg = resolveFfmpegBinary();
    const finalPath = managedOutputPath(root, "reference-cleanup.wav");
    const temporaryPath = temporaryOutputPath(root, "reference-cleanup", ".wav");
    try {
      await runControlledProcess({
        executable: ffmpeg,
        args: buildCleanupFfmpegArgs({ inputPath: input, outputWavPath: temporaryPath }),
        cwd: root,
        label: "Reference cleanup",
        hooks: options,
      });
      await validateProducedFile(root, temporaryPath, "Reference cleanup output");
      await atomicReplaceManagedFile(root, temporaryPath, finalPath);
      return finalPath;
    } finally {
      await fsPromises.rm(temporaryPath, { force: true }).catch(() => undefined);
    }
  }

  const audioSr = resolveAudioSrBinary();
  const temporaryDirectory = managedOutputPath(root, `.audiosr-${randomUUID()}`);
  const finalPath = managedOutputPath(root, "reference-audiosr.wav");
  await fsPromises.mkdir(temporaryDirectory, { recursive: false, mode: 0o700 });
  try {
    const initialEntries = await fsPromises.readdir(temporaryDirectory);
    if (initialEntries.length !== 0) {
      throw new Error("AudioSR managed output directory was not empty before launch.");
    }
    const audioSrStartedAt = Date.now();
    await runControlledProcess({
      executable: audioSr,
      args: buildAudioSrArgs({
        inputPath: input,
        outputDirectory: temporaryDirectory,
        audioSr: {
          model: options.audioSrModel ?? options.audioSr?.model,
          // HTTP inputs arrive as strings; buildAudioSrArgs performs the
          // allow-list validation before this value reaches a subprocess.
          device: (options.audioSrDevice ?? options.audioSr?.device) as
            | AudioSrDevice
            | undefined,
          steps: options.audioSrDdimSteps ?? options.audioSr?.steps,
          guidance: options.audioSrGuidanceScale ?? options.audioSr?.guidance,
          seed: options.audioSrSeed ?? options.audioSr?.seed,
        },
      }),
      cwd: root,
      label: "AudioSR",
      hooks: options,
    });
    const produced = await findAudioSrOutput(temporaryDirectory, audioSrStartedAt);
    await validateProducedFile(temporaryDirectory, produced, "AudioSR output");
    await atomicReplaceManagedFile(root, produced, finalPath);
    return finalPath;
  } finally {
    await fsPromises
      .rm(temporaryDirectory, { recursive: true, force: true })
      .catch(() => undefined);
  }
}

export async function readWavSampleRate(inputPath: string): Promise<number> {
  const handle = await fsPromises.open(inputPath, "r");
  try {
    const stat = await handle.stat();
    const headerLength = Math.min(stat.size, MAX_WAV_HEADER_BYTES);
    if (headerLength < 12) throw new Error("WAV output has an incomplete header.");
    const header = Buffer.alloc(headerLength);
    const { bytesRead } = await handle.read(header, 0, headerLength, 0);
    if (
      bytesRead < 12 ||
      !["RIFF", "RF64"].includes(header.toString("ascii", 0, 4)) ||
      header.toString("ascii", 8, 12) !== "WAVE"
    ) {
      throw new Error("WAV output does not contain a supported RIFF/WAVE header.");
    }

    let offset = 12;
    while (offset + 8 <= bytesRead) {
      const chunkId = header.toString("ascii", offset, offset + 4);
      const chunkSize = header.readUInt32LE(offset + 4);
      const chunkDataOffset = offset + 8;
      if (chunkId === "fmt ") {
        if (chunkSize < 16 || chunkDataOffset + 16 > bytesRead) {
          throw new Error("WAV output has an incomplete fmt chunk.");
        }
        return requireSafeInteger(
          header.readUInt32LE(chunkDataOffset + 4),
          "WAV sample rate",
          1,
          MAX_SAMPLE_RATE
        );
      }
      const nextOffset = chunkDataOffset + chunkSize + (chunkSize % 2);
      if (!Number.isSafeInteger(nextOffset) || nextOffset <= offset) break;
      offset = nextOffset;
    }
    throw new Error("WAV output does not contain a readable fmt chunk.");
  } finally {
    await handle.close();
  }
}

export async function finalizeSpeechAudio(
  options: FinalizeSpeechAudioOptions
): Promise<FinalizedSpeechAudio> {
  const root = await resolveManagedWorkingDirectory(options.workingDir);
  const inputWavPath = await resolveManagedFile(
    root,
    options.inputWavPath,
    "inputWavPath"
  );
  if (path.extname(inputWavPath).toLowerCase() !== ".wav") {
    throw new Error("inputWavPath must end in .wav.");
  }
  if (options.outputFormat !== "wav" && options.outputFormat !== "mp3") {
    throw new Error("outputFormat must be wav or mp3.");
  }
  await validateProducedFile(root, inputWavPath, "WAV output");

  const manifest = options.chapterManifestPath
    ? await loadManagedManifest(root, options.chapterManifestPath)
    : undefined;
  const metadata = manifest
    ? buildFfmetadataDocument(manifest)
    : { text: ";FFMETADATA1\n", chapterCount: 0 };
  const normalizeLevels = options.normalizeLevels === true;
  const sampleRate = normalizeLevels
    ? await readWavSampleRate(inputWavPath)
    : undefined;
  options.assertActive?.();

  if (options.outputFormat === "wav") {
    if (normalizeLevels) {
      const ffmpeg = resolveFfmpegBinary();
      const outputPath = managedOutputPath(
        root,
        `${path.basename(inputWavPath, path.extname(inputWavPath))}-normalized.wav`
      );
      const temporaryOutput = temporaryOutputPath(
        root,
        path.basename(outputPath, ".wav"),
        ".wav"
      );
      try {
        await runControlledProcess({
          executable: ffmpeg,
          args: buildLevelNormalizedWavFfmpegArgs({
            inputWavPath,
            outputWavPath: temporaryOutput,
            sampleRate: sampleRate!,
          }),
          cwd: root,
          label: "Speech level normalization",
          hooks: options,
        });
        await validateProducedFile(root, temporaryOutput, "Normalized WAV output");
        await atomicReplaceManagedFile(root, temporaryOutput, outputPath);
        return {
          outputPath,
          format: "wav",
          mimeType: "audio/wav",
          chapterCount: metadata.chapterCount,
        };
      } finally {
        await fsPromises.rm(temporaryOutput, { force: true }).catch(() => undefined);
      }
    }
    return {
      outputPath: inputWavPath,
      format: "wav",
      mimeType: "audio/wav",
      chapterCount: metadata.chapterCount,
    };
  }

  const ffmpeg = resolveFfmpegBinary();
  const quality = normalizeMp3Quality(options.mp3Quality);
  const outputPath = managedOutputPath(
    root,
    `${path.basename(inputWavPath, path.extname(inputWavPath))}.mp3`
  );
  const temporaryOutput = temporaryOutputPath(
    root,
    path.basename(outputPath, ".mp3"),
    ".mp3"
  );
  const metadataPath =
    metadata.chapterCount > 0
      ? temporaryOutputPath(root, "chapters", ".ffmeta")
      : undefined;

  try {
    if (metadataPath) {
      await fsPromises.writeFile(metadataPath, metadata.text, {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600,
      });
    }
    await runControlledProcess({
      executable: ffmpeg,
      args: buildMp3FfmpegArgs({
        inputWavPath,
        outputMp3Path: temporaryOutput,
        ffmetadataPath: metadataPath,
        mp3Quality: quality,
        normalizeLevels,
        sampleRate,
      }),
      cwd: root,
      label: "MP3 encoding",
      hooks: options,
    });
    await validateProducedFile(root, temporaryOutput, "MP3 output");
    await atomicReplaceManagedFile(root, temporaryOutput, outputPath);
    return {
      outputPath,
      format: "mp3",
      mimeType: "audio/mpeg",
      chapterCount: metadata.chapterCount,
    };
  } finally {
    await Promise.all([
      fsPromises.rm(temporaryOutput, { force: true }).catch(() => undefined),
      metadataPath
        ? fsPromises.rm(metadataPath, { force: true }).catch(() => undefined)
        : Promise.resolve(),
    ]);
  }
}
