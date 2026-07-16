import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import { EventEmitter } from "events";
import { spawn, type ChildProcess } from "child_process";
import readline from "readline";
import { fileURLToPath } from "url";
import { nanoid } from "nanoid";
import type {
  TtsDownloadStatus,
  TtsJobStatus,
  TtsStatus,
  TtsWsMessage,
} from "@shared/schema";
import { terminateChildProcess } from "./process-utils";

type InternalDownloadStatus = TtsDownloadStatus;

type JobState = "queued" | "running" | "completed" | "failed" | "cancelled";

interface InternalJob {
  id: string;
  status: JobState;
  progress: number;
  message?: string;
  outputFile?: string;
  voiceFileName?: string;
  textFileName?: string;
  error?: string;
  createdAt: number;
  updatedAt: number;
  workingDir: string;
}

interface PythonMessage {
  event: string;
  progress?: number;
  message?: string;
  level?: "info" | "warn" | "error";
  output_path?: string;
  error?: string;
}

const OFFICIAL_MODEL_REPO_ID = "IndexTeam/IndexTTS-2";
const OFFICIAL_MODEL_REVISION = "740dcaff396282ffb241903d150ac011cd4b1ede";
const MODEL_MANIFEST_NAME = ".voiceforge-index-model.json";
const MODEL_MANIFEST_VERSION = 2;
const MINIMUM_MODEL_BYTES = 100 * 1024 * 1024;
const MODEL_WEIGHT_EXTENSIONS = new Set([".bin", ".onnx", ".pt", ".pth", ".safetensors"]);
const CONFIGURED_INDEX_PYTHON = process.env.INDEX_TTS_PYTHON?.trim();
// The default interpreter is used only for model-data downloads. Loading and
// synthesis require an explicitly configured, isolated IndexTTS runtime.
const PYTHON_BIN =
  CONFIGURED_INDEX_PYTHON || (process.platform === "win32" ? "python" : "python3");
const INDEX_RUNTIME_SETUP_MESSAGE =
  "IndexTTS runtime is not configured. Stop VoiceForge, run 'VoiceForge.cmd setup-index', " +
  "restart VoiceForge, then select Verify runtime.";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export type TtsLogLevel = "info" | "warn" | "error";

interface ModelManifestFile {
  path: string;
  size: number;
}

interface ModelManifest {
  manifest_version?: unknown;
  repo_id?: unknown;
  revision?: unknown;
  snapshot_path?: unknown;
  files?: unknown;
}

type ModelInspection =
  | { ready: true }
  | { ready: false; missing: boolean; reason: string };

function isInside(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function expectedSnapshotPath(modelsDir: string): string {
  const repositoryCacheName = `models--${OFFICIAL_MODEL_REPO_ID.replace("/", "--")}`;
  return path.resolve(
    modelsDir,
    "hf_cache",
    repositoryCacheName,
    "snapshots",
    OFFICIAL_MODEL_REVISION
  );
}

function collectSnapshotInventory(modelsDir: string, snapshotPath: string): ModelManifestFile[] {
  const inventory: ModelManifestFile[] = [];
  const resolvedModelsDir = fs.realpathSync(modelsDir);
  const visitedDirectories = new Set<string>();

  const visit = (directory: string) => {
    const resolvedDirectory = fs.realpathSync(directory);
    if (!isInside(resolvedModelsDir, resolvedDirectory) || visitedDirectories.has(resolvedDirectory)) {
      if (!isInside(resolvedModelsDir, resolvedDirectory)) {
        throw new Error(`Pinned model directory resolves outside ${modelsDir}`);
      }
      return;
    }
    visitedDirectories.add(resolvedDirectory);

    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const candidate = path.join(directory, entry.name);
      const stat = fs.statSync(candidate);
      const resolvedCandidate = fs.realpathSync(candidate);
      if (!isInside(resolvedModelsDir, resolvedCandidate)) {
        throw new Error(`Pinned model file resolves outside ${modelsDir}: ${candidate}`);
      }
      if (stat.isDirectory()) {
        visit(candidate);
      } else if (stat.isFile()) {
        inventory.push({
          path: path.relative(snapshotPath, candidate).split(path.sep).join("/"),
          size: stat.size,
        });
      }
    }
  };

  visit(snapshotPath);
  inventory.sort((a, b) => a.path.localeCompare(b.path));
  if (inventory.length === 0) throw new Error("The pinned model snapshot contains no files");
  if (!inventory.some((file) => file.path.endsWith("config.yaml") && file.size > 0)) {
    throw new Error("The pinned model snapshot is missing a non-empty config.yaml");
  }
  if (!inventory.some((file) => MODEL_WEIGHT_EXTENSIONS.has(path.extname(file.path).toLowerCase()))) {
    throw new Error("The pinned model snapshot contains no recognized model weights");
  }
  if (inventory.reduce((total, file) => total + file.size, 0) < MINIMUM_MODEL_BYTES) {
    throw new Error("The pinned model snapshot is incomplete (model files are too small)");
  }
  return inventory;
}

function writeModelManifest(modelsDir: string, snapshotPath: string, files: ModelManifestFile[]) {
  const manifestPath = path.join(modelsDir, MODEL_MANIFEST_NAME);
  const temporaryPath = path.join(modelsDir, `${MODEL_MANIFEST_NAME}.tmp`);
  const manifest = {
    manifest_version: MODEL_MANIFEST_VERSION,
    repo_id: OFFICIAL_MODEL_REPO_ID,
    revision: OFFICIAL_MODEL_REVISION,
    snapshot_path: path.relative(modelsDir, snapshotPath).split(path.sep).join("/"),
    files,
  };
  fs.writeFileSync(temporaryPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf-8");
  fs.renameSync(temporaryPath, manifestPath);
}

class IndexTtsService extends EventEmitter {
  private downloadStatus: InternalDownloadStatus = "idle";
  private loadStatus: InternalDownloadStatus = "idle";
  private downloadError?: string;
  private loadError?: string;
  private modelsReady = false;
  private readonly rootDir: string;
  private readonly modelsDir: string;
  private readonly jobsDir: string;
  private readonly workerScript: string;
  private readonly subscribers = new Set<(message: TtsWsMessage) => void>();
  private readonly jobs = new Map<string, InternalJob>();
  private readonly activeProcesses = new Map<string, ChildProcess>();

  constructor(rootDir?: string) {
    super();
    this.rootDir = rootDir || path.join(process.cwd(), "attached_assets", "index-tts");
    this.modelsDir = path.join(this.rootDir, "models");
    this.jobsDir = path.join(this.rootDir, "jobs");
    this.workerScript = this.resolveWorkerScript();
    this.ensureBaseDirs();
    this.refreshModelReadiness();
  }

  private inspectPinnedModels(): ModelInspection {
    const manifestPath = path.join(this.modelsDir, MODEL_MANIFEST_NAME);
    if (!fs.existsSync(manifestPath)) {
      return { ready: false, missing: true, reason: "Pinned model manifest is missing" };
    }

    try {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8")) as ModelManifest;
      if (
        manifest.repo_id !== OFFICIAL_MODEL_REPO_ID ||
        manifest.revision !== OFFICIAL_MODEL_REVISION ||
        typeof manifest.snapshot_path !== "string" ||
        !manifest.snapshot_path
      ) {
        throw new Error("The model manifest does not identify the pinned official snapshot");
      }
      if (path.isAbsolute(manifest.snapshot_path)) {
        throw new Error("The model manifest contains an absolute snapshot path");
      }

      const snapshotPath = path.resolve(this.modelsDir, manifest.snapshot_path);
      if (snapshotPath !== expectedSnapshotPath(this.modelsDir) || !fs.statSync(snapshotPath).isDirectory()) {
        throw new Error("The pinned model snapshot directory is missing or unexpected");
      }
      const actualFiles = collectSnapshotInventory(this.modelsDir, snapshotPath);

      if (manifest.manifest_version === MODEL_MANIFEST_VERSION) {
        if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
          throw new Error("The model manifest has no completeness inventory");
        }
        const expectedFiles = new Map<string, number>();
        for (const rawFile of manifest.files) {
          if (
            typeof rawFile !== "object" ||
            rawFile === null ||
            typeof (rawFile as ModelManifestFile).path !== "string" ||
            !(rawFile as ModelManifestFile).path ||
            !Number.isSafeInteger((rawFile as ModelManifestFile).size) ||
            (rawFile as ModelManifestFile).size < 0 ||
            expectedFiles.has((rawFile as ModelManifestFile).path)
          ) {
            throw new Error("The model manifest contains an invalid file entry");
          }
          expectedFiles.set((rawFile as ModelManifestFile).path, (rawFile as ModelManifestFile).size);
        }
        if (
          expectedFiles.size !== actualFiles.length ||
          actualFiles.some((file) => expectedFiles.get(file.path) !== file.size)
        ) {
          throw new Error("The pinned model snapshot does not match its completeness inventory");
        }
      } else if (manifest.manifest_version === undefined && manifest.files === undefined) {
        // Legacy manifests were written only after the pinned snapshot completed.
        // Persist a complete inventory now so upgrades do not redownload valid weights.
        writeModelManifest(this.modelsDir, snapshotPath, actualFiles);
      } else {
        throw new Error("The model manifest version is unsupported");
      }
      return { ready: true };
    } catch (error) {
      return {
        ready: false,
        missing: false,
        reason: error instanceof Error ? error.message : String(error),
      };
    }
  }

  private refreshModelReadiness(force = false) {
    if (!force && this.downloadStatus === "in-progress") return;
    const inspection = this.inspectPinnedModels();
    this.modelsReady = inspection.ready;
    if (inspection.ready) {
      this.downloadStatus = "completed";
      this.downloadError = undefined;
    } else if (inspection.missing) {
      if (this.downloadStatus === "completed") this.downloadStatus = "idle";
      if (this.downloadStatus !== "failed") this.downloadError = undefined;
    } else {
      this.downloadStatus = "failed";
      this.downloadError = inspection.reason;
    }
  }

  private resolveWorkerScript(): string {
    const distWorker = path.join(__dirname, "python", "index_tts_worker.py");
    const srcWorker = path.join(__dirname, "..", "server", "python", "index_tts_worker.py");

    if (fs.existsSync(distWorker)) {
      return distWorker;
    }

    if (fs.existsSync(srcWorker)) {
      return srcWorker;
    }

    throw new Error("Index TTS worker script not found. Expected at 'server/python/index_tts_worker.py'.");
  }

  private ensureBaseDirs() {
    for (const dir of [this.rootDir, this.modelsDir, this.jobsDir]) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }

  private log(level: TtsLogLevel, message: string) {
    const payload = {
      id: nanoid(),
      level,
      message,
      timestamp: Date.now(),
    };
    this.broadcast({
      type: "log",
      payload,
    });
  }

  private broadcast(message: TtsWsMessage) {
    this.subscribers.forEach((subscriber) => {
      try {
        subscriber(message);
      } catch (err) {
        console.error("TTS subscriber error:", err);
      }
    });
  }

  public subscribe(listener: (message: TtsWsMessage) => void): () => void {
    this.subscribers.add(listener);
    // Send immediate status snapshot
    listener({
      type: "status",
      payload: this.getStatus(),
    });
    return () => {
      this.subscribers.delete(listener);
    };
  }

  private toPublicJob(job: InternalJob): TtsJobStatus {
    return {
      id: job.id,
      status: job.status,
      progress: Math.max(0, Math.min(100, Math.round(job.progress))),
      message: job.message,
      outputFile: job.outputFile,
      voiceFileName: job.voiceFileName,
      textFileName: job.textFileName,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      error: job.error,
    };
  }

  public getStatus(): TtsStatus {
    this.refreshModelReadiness();
    const jobs = Array.from(this.jobs.values())
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((job) => this.toPublicJob(job));
    return {
      downloadStatus: this.downloadStatus,
      loadStatus: this.loadStatus,
      modelsReady: this.modelsReady,
      runtimeConfigured: Boolean(CONFIGURED_INDEX_PYTHON),
      modelsPath: this.modelsDir,
      lastDownloadError: this.downloadError,
      lastLoadError: this.loadError,
      jobs,
    };
  }

  public getJob(jobId: string): TtsJobStatus | undefined {
    const job = this.jobs.get(jobId);
    return job ? this.toPublicJob(job) : undefined;
  }

  public getJobOutputPath(jobId: string): string | undefined {
    const job = this.jobs.get(jobId);
    if (!job?.outputFile || job.status !== "completed") return undefined;
    const resolved = path.resolve(job.outputFile);
    const root = path.resolve(job.workingDir);
    if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) return undefined;
    return resolved;
  }

  public cancelJob(jobId: string): TtsJobStatus | undefined {
    const job = this.jobs.get(jobId);
    if (!job) return undefined;
    if (job.status !== "queued" && job.status !== "running") return this.toPublicJob(job);

    this.updateJob(jobId, {
      status: "cancelled",
      message: "Synthesis cancelled",
      error: undefined,
      outputFile: undefined,
    });
    const child = this.activeProcesses.get(jobId);
    if (child) terminateChildProcess(child);
    this.log("info", `IndexTTS synthesis job ${jobId} cancellation requested`);
    return this.getJob(jobId);
  }

  private updateJob(jobId: string, updates: Partial<InternalJob>) {
    const job = this.jobs.get(jobId);
    if (!job) return;
    if (job.status === "cancelled") return;
    Object.assign(job, updates, { updatedAt: Date.now() });
    this.jobs.set(jobId, job);
    this.broadcast({
      type: "job",
      payload: this.toPublicJob(job),
    });
  }

  private async runPython(
    command: string,
    args: string[],
    onMessage?: (message: PythonMessage) => void,
    jobId?: string
  ): Promise<void> {
    const pythonArgs = [
      "-u",
      this.workerScript,
      "--root-dir",
      this.rootDir,
      "--models-dir",
      this.modelsDir,
      command,
      ...args,
    ];

    const child = spawn(PYTHON_BIN, pythonArgs, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        INDEX_TTS_ROOT: this.rootDir,
        INDEX_TTS_MODELS_DIR: this.modelsDir,
        // IndexTTS prints SentencePiece tokens (including U+2581) while it
        // synthesizes. Windows otherwise gives redirected Python streams the
        // active ANSI code page, which cannot represent those tokens.
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (jobId) this.activeProcesses.set(jobId, child);

    const releaseProcess = () => {
      if (jobId && this.activeProcesses.get(jobId) === child) {
        this.activeProcesses.delete(jobId);
      }
    };

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    const stdout = readline.createInterface({ input: child.stdout });
    const stderrChunks: string[] = [];
    let workerError: string | undefined;

    stdout.on("line", (line) => {
      if (!line) return;
      try {
        const parsed = JSON.parse(line) as PythonMessage;
        if (parsed.event === "error" && parsed.error) workerError = parsed.error;
        onMessage?.(parsed);
      } catch {
        const trimmed = line.trim();
        if (trimmed.length > 0) {
          this.log("info", trimmed);
        }
      }
    });

    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderrChunks.push(text);
      console.error("[IndexTTS python]", text.trim());
    });

    await new Promise<void>((resolve, reject) => {
      child.on("error", (err) => {
        releaseProcess();
        reject(err);
      });
      child.on("exit", releaseProcess);
      child.on("close", (code) => {
        releaseProcess();
        if (code === 0) {
          resolve();
        } else {
          const hint =
            code === 9009
              ? "\n(Hint: On Windows, this often means Python isn’t on PATH or the alias points to the Microsoft Store. Install Python and ensure 'python' works, or set INDEX_TTS_PYTHON)"
              : "";
          const details = workerError?.trim() || stderrChunks.join("\n").trim();
          const error = new Error(
            details || `IndexTTS worker exited with code ${code}.${hint}`
          );
          reject(error);
        }
      });
    });
  }

  public async downloadModels() {
    this.refreshModelReadiness();
    if (this.modelsReady) {
      this.log("info", "Pinned IndexTTS models are already complete; skipping download");
      return;
    }
    if (this.downloadStatus === "in-progress") {
      throw new Error("Download already in progress");
    }

    this.downloadStatus = "in-progress";
    this.downloadError = undefined;
    this.broadcast({
      type: "status",
      payload: this.getStatus(),
    });
    this.log(
      "info",
      `Starting pinned IndexTTS model download from ${OFFICIAL_MODEL_REPO_ID}@${OFFICIAL_MODEL_REVISION}`
    );

    try {
      await this.runPython("download", [], (message) => {
        if (message.event === "progress") {
          this.log("info", message.message ?? "Downloading models…");
        } else if (message.event === "log") {
          this.log(message.level || "info", message.message || "");
        }
      });
      this.refreshModelReadiness(true);
      if (!this.modelsReady) {
        throw new Error(this.downloadError || "Downloaded model snapshot failed completeness validation");
      }
      this.log("info", "IndexTTS models downloaded");
    } catch (error) {
      this.downloadStatus = "failed";
      this.downloadError = error instanceof Error ? error.message : String(error);
      this.log("error", `Model download failed: ${this.downloadError}`);
      throw error;
    } finally {
      this.broadcast({
        type: "status",
        payload: this.getStatus(),
      });
    }
  }

  public async loadModels() {
    this.refreshModelReadiness();
    if (this.loadStatus === "in-progress") {
      throw new Error("Model load already in progress");
    }
    if (!this.modelsReady) {
      throw new Error("Models must be downloaded before loading");
    }
    if (!CONFIGURED_INDEX_PYTHON) {
      this.loadStatus = "failed";
      this.loadError = INDEX_RUNTIME_SETUP_MESSAGE;
      this.log("error", this.loadError);
      this.broadcast({ type: "status", payload: this.getStatus() });
      throw new Error(this.loadError);
    }

    this.loadStatus = "in-progress";
    this.loadError = undefined;
    this.broadcast({
      type: "status",
      payload: this.getStatus(),
    });
    this.log("info", "Loading IndexTTS models into memory");

    try {
      await this.runPython("load", [], (message) => {
        if (message.event === "progress") {
          this.log("info", message.message ?? "Loading models…");
        } else if (message.event === "log") {
          this.log(message.level || "info", message.message || "");
        }
      });
      this.loadStatus = "completed";
      this.log("info", "IndexTTS models loaded");
    } catch (error) {
      this.loadStatus = "failed";
      this.loadError = error instanceof Error ? error.message : String(error);
      this.log("error", `Model loading failed: ${this.loadError}`);
      throw error;
    } finally {
      this.broadcast({
        type: "status",
        payload: this.getStatus(),
      });
    }
  }

  public async startSynthesis(params: {
    voiceBuffer: Buffer;
    voiceFileName: string;
    textContent: string;
    textFileName?: string;
  }): Promise<TtsJobStatus> {
    this.refreshModelReadiness();
    if (!this.modelsReady || this.loadStatus !== "completed") {
      throw new Error("Download the pinned models and verify the IndexTTS runtime before synthesis");
    }

    const jobId = nanoid();
    const jobDir = path.join(this.jobsDir, jobId);
    await fsPromises.mkdir(jobDir, { recursive: true });

    const voiceExt = path.extname(params.voiceFileName || "").toLowerCase();
    const voicePath = path.join(jobDir, `voice${voiceExt || ".wav"}`);
    const textPath = path.join(jobDir, "script.txt");
    const outputPath = path.join(jobDir, "output.wav");

    await fsPromises.writeFile(voicePath, params.voiceBuffer);
    await fsPromises.writeFile(textPath, params.textContent, "utf-8");

    const createdAt = Date.now();
    const job: InternalJob = {
      id: jobId,
      status: "running",
      progress: 5,
      message: "Starting synthesis…",
      voiceFileName: params.voiceFileName,
      textFileName: params.textFileName ?? "script.txt",
      createdAt,
      updatedAt: createdAt,
      workingDir: jobDir,
    };

    this.jobs.set(jobId, job);
    this.broadcast({
      type: "job",
      payload: this.toPublicJob(job),
    });
    this.log("info", `Starting IndexTTS synthesis job ${jobId}`);

    void this.runPython(
      "synthesize",
      [
        "--voice",
        voicePath,
        "--text",
        textPath,
        "--output",
        outputPath,
      ],
      (message) => {
        if (this.jobs.get(jobId)?.status === "cancelled") return;
        if (message.event === "progress") {
          const progress = typeof message.progress === "number" ? message.progress * 100 : job.progress;
          this.updateJob(jobId, {
            progress,
            message: message.message,
          });
        } else if (message.event === "log") {
          this.log(message.level || "info", message.message || "");
        } else if (message.event === "complete") {
          if (!fs.existsSync(outputPath)) {
            this.updateJob(jobId, {
              status: "failed",
              error: "IndexTTS worker did not create output.wav",
              message: "Synthesis output is missing",
              outputFile: undefined,
            });
            this.log("error", `IndexTTS synthesis job ${jobId} did not create output.wav`);
            return;
          }
          this.updateJob(jobId, {
            status: "completed",
            progress: 100,
            message: message.message ?? "Synthesis complete",
            outputFile: outputPath,
          });
        } else if (message.event === "error") {
          this.updateJob(jobId, {
            status: "failed",
            progress: job.progress,
            error: message.error || message.message || "Unknown synthesis error",
            message: message.message,
          });
        }
      },
      jobId
    ).then(
      () => {
        const finalJob = this.jobs.get(jobId);
        if (!finalJob || finalJob.status === "cancelled" || finalJob.status === "failed") return;
        if (!fs.existsSync(outputPath)) {
          this.updateJob(jobId, {
            status: "failed",
            error: "IndexTTS worker exited successfully but output.wav is missing",
            message: "Synthesis output is missing",
            outputFile: undefined,
          });
          this.log("error", `IndexTTS synthesis job ${jobId} exited without output.wav`);
          return;
        }
        if (finalJob.status === "queued" || finalJob.status === "running") {
          this.updateJob(jobId, {
            status: "completed",
            progress: 100,
            message: "Synthesis complete",
            outputFile: outputPath,
          });
        }
        this.log("info", `IndexTTS synthesis job ${jobId} finished`);
      },
      (error) => {
        if (this.jobs.get(jobId)?.status === "cancelled") return;
        this.updateJob(jobId, {
          status: "failed",
          error: error instanceof Error ? error.message : String(error),
          message: "Synthesis failed",
        });
        this.log(
          "error",
          `IndexTTS synthesis job ${jobId} failed: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }
    );

    return this.toPublicJob(job);
  }
}

export const indexTtsService = new IndexTtsService();
