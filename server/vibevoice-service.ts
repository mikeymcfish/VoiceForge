import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import { EventEmitter } from "events";
import { spawn } from "child_process";
import readline from "readline";
import { fileURLToPath } from "url";
import { nanoid } from "nanoid";
import type {
  VibevoiceJobStatus,
  VibevoiceStatus,
  VibevoiceWsMessage,
} from "@shared/schema";

const DEFAULT_REPO_URL =
  process.env.VIBEVOICE_REPO_URL || "https://github.com/vibevoice-community/VibeVoice.git";
const DEFAULT_REPO_BRANCH = process.env.VIBEVOICE_REPO_BRANCH || "main";
const PYTHON_BIN = process.env.VIBEVOICE_PYTHON || "python3";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

type JobState = "queued" | "running" | "completed" | "failed";

type WorkerLogLevel = "info" | "warn" | "error";

interface InternalJob {
  id: string;
  status: JobState;
  progress: number;
  message?: string;
  outputFile?: string;
  voiceFileName?: string;
  textFileName?: string;
  style?: string;
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

  constructor(rootDir?: string) {
    super();
    this.rootDir = rootDir || path.join(process.cwd(), "attached_assets", "vibevoice");
    this.repoDir = path.join(this.rootDir, "repo");
    this.jobsDir = path.join(this.rootDir, "jobs");
    this.workerScript = this.resolveWorkerScript();
    this.ensureBaseDirs();
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
      textFileName: job.textFileName,
      style: job.style,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      error: job.error,
    };
  }

  public getStatus(): VibevoiceStatus {
    const jobs = Array.from(this.jobs.values())
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((job) => this.toPublicJob(job));

    return {
      setupStatus: this.setupStatus,
      ready: this.setupStatus === "completed",
      repoPath: this.repoDir,
      lastSetupError: this.setupError,
      jobs,
    };
  }

  private updateJob(id: string, patch: Partial<InternalJob>) {
    const job = this.jobs.get(id);
    if (!job) return;

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
    onMessage?: (message: WorkerMessage) => void
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
          env: {
            ...process.env,
          },
        }
      );

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
        stdoutRl.close();
        reject(error);
      });

      child.on("close", (code) => {
        stdoutRl.close();
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`VibeVoice worker exited with code ${code}`));
        }
      });
    });
  }

  public async startSetup(repoUrl?: string, branch?: string): Promise<void> {
    if (this.setupStatus === "in-progress") {
      throw new Error("Setup already in progress");
    }

    const targetRepo = repoUrl && repoUrl.trim().length > 0 ? repoUrl.trim() : DEFAULT_REPO_URL;
    const targetBranch = branch && branch.trim().length > 0 ? branch.trim() : DEFAULT_REPO_BRANCH;

    this.setupStatus = "in-progress";
    this.setupError = undefined;
    this.broadcast({
      type: "status",
      payload: this.getStatus(),
    });

    try {
      await this.runPython(
        "setup",
        ["--repo-url", targetRepo, "--repo-branch", targetBranch],
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
    voiceBuffer?: Buffer;
    voiceFileName?: string;
    textContent: string;
    textFileName?: string;
    style?: string;
    temperature?: number;
  }): Promise<VibevoiceJobStatus> {
    if (this.setupStatus !== "completed") {
      throw new Error("Run setup before starting synthesis");
    }

    const jobId = nanoid();
    const jobDir = path.join(this.jobsDir, jobId);
    await fsPromises.mkdir(jobDir, { recursive: true });

    let voicePath: string | undefined;
    if (params.voiceBuffer && params.voiceBuffer.length > 0) {
      const ext = params.voiceFileName ? path.extname(params.voiceFileName) : ".wav";
      voicePath = path.join(jobDir, `voice${ext || ".wav"}`);
      await fsPromises.writeFile(voicePath, params.voiceBuffer);
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
      textFileName: params.textFileName ?? "script.txt",
      style: params.style,
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

    if (voicePath) {
      args.push("--voice", voicePath);
    }
    if (params.style) {
      args.push("--style", params.style);
    }
    if (typeof params.temperature === "number" && Number.isFinite(params.temperature)) {
      args.push("--temperature", String(params.temperature));
    }

    void this.runPython(
      "synthesize",
      args,
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
            outputFile: message.output_path || outputPath,
          });
        } else if (message.event === "error") {
          this.updateJob(jobId, {
            status: "failed",
            error: message.error || message.message || "Unknown synthesis error",
            message: message.message,
          });
        }
      }
    ).then(
      () => {
        const finalJob = this.jobs.get(jobId);
        if (finalJob && finalJob.status !== "completed") {
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
}

export const vibevoiceService = new VibevoiceService();
