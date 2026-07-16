import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import { EventEmitter } from "events";
import { spawn, type ChildProcess } from "child_process";
import readline from "readline";
import { fileURLToPath } from "url";
import { nanoid } from "nanoid";
import type {
  VibevoiceJobStatus,
  VibevoiceStatus,
  VibevoiceWsMessage,
} from "@shared/schema";
import { terminateChildProcess } from "./process-utils";

// Prefer a Windows-friendly default. Users can override via VIBEVOICE_PYTHON
const PYTHON_BIN =
  process.env.VIBEVOICE_PYTHON || (process.platform === "win32" ? "python" : "python3");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

type JobState = "queued" | "running" | "completed" | "failed" | "cancelled";

type WorkerLogLevel = "info" | "warn" | "error";

const PINNED_MODEL_REVISIONS: Record<string, string> = {
  "microsoft/VibeVoice-1.5B": "c00898d257e6b46004e3e2866a47534085fb685a",
  "aoi-ot/VibeVoice-Large": "8229be00d7c036aa32321e4dae8a81d433f6413a",
};
const PINNED_REPO_REVISION = "07cb79feadd2d3fd7f47530d4c964a12857936a0";

interface InternalJob {
  id: string;
  status: JobState;
  progress: number;
  message?: string;
  outputFile?: string;
  voiceFileName?: string;
  voiceFileNames?: string[];
  textFileName?: string;
  style?: string;
  selectedModel?: string;
  error?: string;
  createdAt: number;
  updatedAt: number;
  workingDir: string;
}

interface WorkerMessage {
  event: string;
  progress?: number;
  message?: string;
  level?: WorkerLogLevel;
  output_path?: string;
  error?: string;
}

export type VibevoiceLogLevel = WorkerLogLevel;

class VibevoiceService extends EventEmitter {
  private setupStatus: VibevoiceStatus["setupStatus"] = "idle";
  private setupError?: string;
  private readonly rootDir: string;
  private readonly repoDir: string;
  private readonly jobsDir: string;
  private readonly workerScript: string;
  private readonly subscribers = new Set<(message: VibevoiceWsMessage) => void>();
  private readonly jobs = new Map<string, InternalJob>();
  private readonly activeProcesses = new Map<string, ChildProcess>();

  constructor(rootDir?: string) {
    super();
    this.rootDir = rootDir || path.join(process.cwd(), "attached_assets", "vibevoice");
    this.repoDir = path.join(this.rootDir, "repo");
    this.jobsDir = path.join(this.rootDir, "jobs");
    this.workerScript = this.resolveWorkerScript();
    this.ensureBaseDirs();
    if (this.isPinnedRepoReady() && this.listAvailableModels().length > 0) {
      this.setupStatus = "completed";
    }
  }

  private isPinnedRepoReady(): boolean {
    try {
      const gitHead = fs.readFileSync(path.join(this.repoDir, ".git", "HEAD"), "utf-8").trim();
      return (
        gitHead.toLowerCase() === PINNED_REPO_REVISION &&
        fs.existsSync(path.join(this.repoDir, "pyproject.toml"))
      );
    } catch {
      return false;
    }
  }

  private resolveWorkerScript(): string {
    const distWorker = path.join(__dirname, "python", "vibevoice_worker.py");
    const srcWorker = path.join(__dirname, "..", "server", "python", "vibevoice_worker.py");

    if (fs.existsSync(distWorker)) {
      return distWorker;
    }

    if (fs.existsSync(srcWorker)) {
      return srcWorker;
    }

    throw new Error("VibeVoice worker script not found. Expected at 'server/python/vibevoice_worker.py'.");
  }

  private ensureBaseDirs() {
    for (const dir of [this.rootDir, this.repoDir, this.jobsDir]) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }

  private log(level: WorkerLogLevel, message: string) {
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

  private broadcast(message: VibevoiceWsMessage) {
    this.subscribers.forEach((subscriber) => {
      try {
        subscriber(message);
      } catch (err) {
        console.error("VibeVoice subscriber error:", err);
      }
    });
  }

  public subscribe(listener: (message: VibevoiceWsMessage) => void): () => void {
    this.subscribers.add(listener);
    listener({
      type: "status",
      payload: this.getStatus(),
    });
    return () => {
      this.subscribers.delete(listener);
    };
  }

  private toPublicJob(job: InternalJob): VibevoiceJobStatus {
    return {
      id: job.id,
      status: job.status,
      progress: Math.max(0, Math.min(100, Math.round(job.progress))),
      message: job.message,
      outputFile: job.outputFile,
      voiceFileName: job.voiceFileName,
      voiceFileNames: job.voiceFileNames,
      textFileName: job.textFileName,
      style: job.style,
      selectedModel: job.selectedModel,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      error: job.error,
    };
  }

  public getStatus(): VibevoiceStatus {
    const jobs = Array.from(this.jobs.values())
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((job) => this.toPublicJob(job));

    const availableModels = this.listAvailableModels();
    return {
      setupStatus: this.setupStatus,
      ready:
        this.setupStatus === "completed" &&
        this.isPinnedRepoReady() &&
        availableModels.length > 0,
      repoPath: this.repoDir,
      lastSetupError: this.setupError,
      availableModels,
      jobs,
    };
  }

  private listAvailableModels(): { id: string; path: string }[] {
    const modelsRoot = path.join(this.rootDir, "models");
    if (!fs.existsSync(modelsRoot)) return [];
    const entries = fs.readdirSync(modelsRoot, { withFileTypes: true });
    const models: { id: string; path: string }[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const dirPath = path.join(modelsRoot, entry.name);
      try {
        const marker = JSON.parse(
          fs.readFileSync(path.join(dirPath, "voiceforge-model.json"), "utf-8")
        ) as {
          complete?: boolean;
          repo_id?: string;
          revision?: string | null;
          artifacts?: { path?: string; size?: number }[];
        };
        const id = marker.repo_id?.trim();
        if (!marker.complete || !id || !Array.isArray(marker.artifacts) || marker.artifacts.length === 0) {
          continue;
        }
        const pinnedRevision = PINNED_MODEL_REVISIONS[id];
        if (pinnedRevision && marker.revision !== pinnedRevision) continue;
        const root = path.resolve(dirPath);
        const valid = marker.artifacts.every((artifact) => {
          if (!artifact.path || !Number.isSafeInteger(artifact.size) || (artifact.size ?? 0) <= 0) return false;
          const artifactPath = path.resolve(root, artifact.path);
          if (artifactPath !== root && !artifactPath.startsWith(`${root}${path.sep}`)) return false;
          try {
            const stat = fs.statSync(artifactPath);
            return stat.isFile() && stat.size === artifact.size;
          } catch {
            return false;
          }
        });
        if (valid) models.push({ id, path: dirPath });
      } catch {
        // A missing or invalid completion manifest means the snapshot is not selectable.
      }
    }
    models.sort((a, b) => a.id.localeCompare(b.id));
    return models;
  }

  private updateJob(id: string, patch: Partial<InternalJob>) {
    const job = this.jobs.get(id);
    if (!job) return;
    if (job.status === "cancelled") return;

    const updated: InternalJob = {
      ...job,
      ...patch,
      updatedAt: Date.now(),
    };
    this.jobs.set(id, updated);
    this.broadcast({
      type: "job",
      payload: this.toPublicJob(updated),
    });
  }

  private runPython(
    command: string,
    args: string[],
    onMessage?: (message: WorkerMessage) => void,
    jobId?: string
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const child = spawn(
        PYTHON_BIN,
        [
          this.workerScript,
          "--root-dir",
          this.rootDir,
          "--repo-dir",
          this.repoDir,
          "--jobs-dir",
          this.jobsDir,
          command,
          ...args,
        ],
        {
          cwd: this.rootDir,
          detached: process.platform !== "win32",
          env: {
            ...process.env,
          },
        }
      );
      if (jobId) this.activeProcesses.set(jobId, child);

      const releaseProcess = () => {
        if (jobId && this.activeProcesses.get(jobId) === child) {
          this.activeProcesses.delete(jobId);
        }
      };

      child.stdout.setEncoding("utf-8");
      const stdoutRl = readline.createInterface({ input: child.stdout });
      stdoutRl.on("line", (line) => {
        if (!line.trim()) return;
        try {
          const message = JSON.parse(line) as WorkerMessage;
          onMessage?.(message);
        } catch (error) {
          this.log("info", line);
        }
      });

      child.stderr.setEncoding("utf-8");
      child.stderr.on("data", (data: string) => {
        data
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .forEach((line) => this.log("warn", line));
      });

      child.on("error", (error) => {
        releaseProcess();
        stdoutRl.close();
        reject(error);
      });
      child.on("exit", releaseProcess);

      child.on("close", (code) => {
        releaseProcess();
        stdoutRl.close();
        if (code === 0) {
          resolve();
        } else {
          const hint =
            code === 9009
              ? " (Windows 'command not found'. Ensure Python is installed and on PATH, or set VIBEVOICE_PYTHON)"
              : "";
          reject(new Error(`VibeVoice worker exited with code ${code}${hint}`));
        }
      });
    });
  }

  public async startSetup(): Promise<void> {
    if (this.setupStatus === "in-progress") {
      throw new Error("Setup already in progress");
    }

    this.setupStatus = "in-progress";
    this.setupError = undefined;
    this.broadcast({
      type: "status",
      payload: this.getStatus(),
    });

    try {
      await this.runPython(
        "setup",
        [],
        (message) => {
          if (message.event === "progress") {
            if (typeof message.message === "string") {
              this.log("info", message.message);
            }
          } else if (message.event === "log") {
            this.log(message.level || "info", message.message || "");
          }
        }
      );
      this.setupStatus = "completed";
      this.log("info", "VibeVoice setup completed");
    } catch (error) {
      this.setupStatus = "failed";
      this.setupError = error instanceof Error ? error.message : String(error);
      this.log("error", `VibeVoice setup failed: ${this.setupError}`);
      throw error;
    } finally {
      this.broadcast({
        type: "status",
        payload: this.getStatus(),
      });
    }
  }

  public async startSynthesis(params: {
    voiceBuffers?: Buffer[];
    voiceFileNames?: (string | undefined)[];
    voiceBuffer?: Buffer; // backwards-compat
    voiceFileName?: string; // backwards-compat
    textContent: string;
    textFileName?: string;
    style?: string;
    guidanceScale?: number;
    modelId?: string;
  }): Promise<VibevoiceJobStatus> {
    if (!this.getStatus().ready) {
      throw new Error("Run setup before starting synthesis");
    }

    const jobId = nanoid();
    const jobDir = path.join(this.jobsDir, jobId);
    await fsPromises.mkdir(jobDir, { recursive: true });

    const voicePaths: string[] = [];
    const voiceFiles: string[] = [];
    const buffers: Buffer[] = [];
    const names: (string | undefined)[] = [];
    if (params.voiceBuffers && params.voiceBuffers.length > 0) {
      buffers.push(...params.voiceBuffers);
      if (params.voiceFileNames) names.push(...params.voiceFileNames);
    } else if (params.voiceBuffer && params.voiceBuffer.length > 0) {
      buffers.push(params.voiceBuffer);
      names.push(params.voiceFileName);
    }
    for (let i = 0; i < Math.min(4, buffers.length); i++) {
      const buf = buffers[i];
      if (!buf || buf.length === 0) continue;
      const orig = names[i];
      const ext = orig ? path.extname(orig) : ".wav";
      const p = path.join(jobDir, `voice${i + 1}${ext || ".wav"}`);
      await fsPromises.writeFile(p, buf);
      voicePaths.push(p);
      if (orig) voiceFiles.push(orig);
    }

    const textPath = path.join(jobDir, "script.txt");
    await fsPromises.writeFile(textPath, params.textContent, "utf-8");

    const outputPath = path.join(jobDir, "output.wav");

    const createdAt = Date.now();
    const job: InternalJob = {
      id: jobId,
      status: "running",
      progress: 5,
      message: "Starting synthesis…",
      voiceFileName: params.voiceFileName,
      voiceFileNames: voiceFiles.length > 0 ? voiceFiles : undefined,
      textFileName: params.textFileName ?? "script.txt",
      style: params.style,
      selectedModel: params.modelId,
      createdAt,
      updatedAt: createdAt,
      workingDir: jobDir,
    };

    this.jobs.set(jobId, job);
    this.broadcast({
      type: "job",
      payload: this.toPublicJob(job),
    });
    this.log("info", `Starting VibeVoice synthesis job ${jobId}`);

    const args: string[] = [
      "--job-id",
      jobId,
      "--text",
      textPath,
      "--output",
      outputPath,
    ];

    for (const vp of voicePaths) {
      args.push("--voice", vp);
    }
    if (params.style) {
      args.push("--style", params.style);
    }
    if (typeof params.guidanceScale === "number" && Number.isFinite(params.guidanceScale)) {
      args.push("--guidance-scale", String(params.guidanceScale));
    }
    if (params.modelId) {
      args.push("--model-id", params.modelId);
    }

    void this.runPython(
      "synthesize",
      args,
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
              error: "VibeVoice worker did not create output.wav",
              message: "Synthesis output is missing",
              outputFile: undefined,
            });
            this.log("error", `VibeVoice synthesis job ${jobId} did not create output.wav`);
            return;
          }
          this.updateJob(jobId, {
            status: "completed",
            progress: 100,
            message: message.message ?? "Synthesis complete",
            outputFile: message.output_path || outputPath,
          });
        } else if (message.event === "error") {
          this.updateJob(jobId, {
            status: "failed",
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
            error: "VibeVoice worker exited successfully but output.wav is missing",
            message: "Synthesis output is missing",
            outputFile: undefined,
          });
          this.log("error", `VibeVoice synthesis job ${jobId} exited without output.wav`);
          return;
        }
        if (finalJob.status === "queued" || finalJob.status === "running") {
          const maybeOutput = finalJob.outputFile || outputPath;
          this.updateJob(jobId, {
            status: "completed",
            progress: 100,
            message: finalJob.message || "Synthesis complete",
            outputFile: maybeOutput,
          });
        }
        this.log("info", `VibeVoice synthesis job ${jobId} finished`);
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
          `VibeVoice synthesis job ${jobId} failed: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    );

    return this.toPublicJob(job);
  }

  public getJob(id: string): VibevoiceJobStatus | undefined {
    const job = this.jobs.get(id);
    return job ? this.toPublicJob(job) : undefined;
  }

  public getJobOutputPath(id: string): string | undefined {
    const job = this.jobs.get(id);
    if (!job?.outputFile || job.status !== "completed") return undefined;
    const resolved = path.resolve(job.outputFile);
    const root = path.resolve(job.workingDir);
    if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) return undefined;
    return resolved;
  }

  public cancelJob(id: string): VibevoiceJobStatus | undefined {
    const job = this.jobs.get(id);
    if (!job) return undefined;
    if (job.status !== "queued" && job.status !== "running") return this.toPublicJob(job);

    this.updateJob(id, {
      status: "cancelled",
      message: "Synthesis cancelled",
      error: undefined,
      outputFile: undefined,
    });
    const child = this.activeProcesses.get(id);
    if (child) terminateChildProcess(child, { processGroup: process.platform !== "win32" });
    this.log("info", `VibeVoice synthesis job ${id} cancellation requested`);
    return this.getJob(id);
  }
}

export const vibevoiceService = new VibevoiceService();
