import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import { EventEmitter } from "events";
import { spawn } from "child_process";
import readline from "readline";
import { fileURLToPath } from "url";
import { nanoid } from "nanoid";
import type {
  TtsDownloadStatus,
  TtsJobStatus,
  TtsStatus,
  TtsWsMessage,
} from "@shared/schema";

type InternalDownloadStatus = TtsDownloadStatus;

type JobState = "queued" | "running" | "completed" | "failed";

interface InternalJob {
  id: string;
  status: JobState;
  progress: number;
  message?: string;
  steps?: number;
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

const DEFAULT_REPO_ID = process.env.INDEX_TTS_REPO || "IndexTeam/IndexTTS-2";
const PYTHON_BIN = process.env.INDEX_TTS_PYTHON || "python3";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export type TtsLogLevel = "info" | "warn" | "error";

class IndexTtsService extends EventEmitter {
  private downloadStatus: InternalDownloadStatus = "idle";
  private loadStatus: InternalDownloadStatus = "idle";
  private downloadError?: string;
  private loadError?: string;
  private readonly rootDir: string;
  private readonly modelsDir: string;
  private readonly jobsDir: string;
  private readonly workerScript: string;
  private readonly subscribers = new Set<(message: TtsWsMessage) => void>();
  private readonly jobs = new Map<string, InternalJob>();

  constructor(rootDir?: string) {
    super();
    this.rootDir = rootDir || path.join(process.cwd(), "attached_assets", "index-tts");
    this.modelsDir = path.join(this.rootDir, "models");
    this.jobsDir = path.join(this.rootDir, "jobs");
    this.workerScript = path.join(__dirname, "python", "index_tts_worker.py");
    this.ensureBaseDirs();
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
      steps: job.steps,
      outputFile: job.outputFile,
      voiceFileName: job.voiceFileName,
      textFileName: job.textFileName,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      error: job.error,
    };
  }

  public getStatus(): TtsStatus {
    const jobs = Array.from(this.jobs.values())
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((job) => this.toPublicJob(job));
    return {
      downloadStatus: this.downloadStatus,
      loadStatus: this.loadStatus,
      modelsReady: this.downloadStatus === "completed" && this.loadStatus === "completed",
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
    return job?.outputFile;
  }

  private updateJob(jobId: string, updates: Partial<InternalJob>) {
    const job = this.jobs.get(jobId);
    if (!job) return;
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
    onMessage?: (message: PythonMessage) => void
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
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    const stdout = readline.createInterface({ input: child.stdout });
    const stderrChunks: string[] = [];

    stdout.on("line", (line) => {
      if (!line) return;
      try {
        const parsed = JSON.parse(line) as PythonMessage;
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
      child.on("error", (err) => reject(err));
      child.on("close", (code) => {
        if (code === 0) {
          resolve();
        } else {
          const error = new Error(
            `IndexTTS worker exited with code ${code}\n${stderrChunks.join("\n")}`
          );
          reject(error);
        }
      });
    });
  }

  public async downloadModels(repoId?: string) {
    const resolvedRepo =
      typeof repoId === "string" && repoId.trim().length > 0 ? repoId : DEFAULT_REPO_ID;
    if (this.downloadStatus === "in-progress") {
      throw new Error("Download already in progress");
    }

    this.downloadStatus = "in-progress";
    this.downloadError = undefined;
    this.broadcast({
      type: "status",
      payload: this.getStatus(),
    });
    this.log("info", `Starting IndexTTS model download from ${resolvedRepo}`);

    try {
      await this.runPython("download", ["--repo-id", resolvedRepo], (message) => {
        if (message.event === "progress") {
          this.log("info", message.message ?? "Downloading models…");
        } else if (message.event === "log") {
          this.log(message.level || "info", message.message || "");
        }
      });
      this.downloadStatus = "completed";
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
    if (this.loadStatus === "in-progress") {
      throw new Error("Model load already in progress");
    }
    if (this.downloadStatus !== "completed") {
      throw new Error("Models must be downloaded before loading");
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
    steps: number;
  }): Promise<TtsJobStatus> {
    if (this.downloadStatus !== "completed") {
      throw new Error("Models must be downloaded before synthesis");
    }

    const steps = Math.min(50, Math.max(20, Math.round(params.steps)));
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
      steps,
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
        "--steps",
        String(steps),
      ],
      (message) => {
        if (message.event === "progress") {
          const progress = typeof message.progress === "number" ? message.progress * 100 : job.progress;
          this.updateJob(jobId, {
            progress,
            message: message.message,
          });
        } else if (message.event === "log") {
          this.log(message.level || "info", message.message || "");
        } else if (message.event === "complete") {
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
      }
    ).then(
      () => {
        const finalJob = this.jobs.get(jobId);
        if (finalJob && finalJob.status !== "completed") {
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
