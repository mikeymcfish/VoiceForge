import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import { EventEmitter } from "events";
import { spawn } from "child_process";
import readline from "readline";
import { fileURLToPath } from "url";
import { nanoid } from "nanoid";
import type {
  PdfOcrConfig,
  PdfOcrDownloadStatus,
  PdfOcrJobStatus,
  PdfOcrLogEntry,
  PdfOcrStatus,
  PdfOcrWsMessage,
} from "@shared/schema";

const DEFAULT_PYTHON_BIN =
  process.env.PDF_OCR_PYTHON || (process.platform === "win32" ? "python" : "python3");

type PersistedPdfOcrConfig = Pick<
  PdfOcrConfig,
  "pythonPath" | "deepseekRepoPath" | "huggingFaceRepoId" | "huggingFaceRevision"
>;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface InternalJob {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message?: string;
  pageCount?: number;
  processedPages?: number;
  outputFile?: string;
  pdfFileName?: string;
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
  processed_pages?: number;
  total_pages?: number;
  output_path?: string;
  error?: string;
  text?: string;
  models_dir?: string;
  deepseek_module_path?: string;
}

class PdfOcrService extends EventEmitter {
  private readonly rootDir: string;
  private readonly modelsDir: string;
  private readonly jobsDir: string;
  private readonly workerScript: string;
  private readonly configPath: string;
  private readonly subscribers = new Set<(message: PdfOcrWsMessage) => void>();
  private readonly jobs = new Map<string, InternalJob>();
  private config: PersistedPdfOcrConfig = {};
  private lastResolvedModulePath?: string;
  private modelDownloadStatus: PdfOcrDownloadStatus = "idle";
  private modelDownloadError?: string;

  constructor(rootDir?: string) {
    super();
    this.rootDir = rootDir || path.join(process.cwd(), "attached_assets", "pdf-ocr");
    this.modelsDir = path.join(this.rootDir, "models");
    this.jobsDir = path.join(this.rootDir, "jobs");
    this.workerScript = this.resolveWorkerScript();
    this.configPath = path.join(this.rootDir, "config.json");
    this.ensureBaseDirs();
    this.loadConfig();
  }

  private resolveWorkerScript(): string {
    const distWorker = path.join(__dirname, "python", "pdf_ocr_worker.py");
    const srcWorker = path.join(__dirname, "..", "server", "python", "pdf_ocr_worker.py");

    if (fs.existsSync(distWorker)) {
      return distWorker;
    }

    if (fs.existsSync(srcWorker)) {
      return srcWorker;
    }

    throw new Error("PDF OCR worker script not found. Expected at 'server/python/pdf_ocr_worker.py'.");
  }

  private ensureBaseDirs() {
    for (const dir of [this.rootDir, this.modelsDir, this.jobsDir]) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }

  private loadConfig() {
    try {
      if (fs.existsSync(this.configPath)) {
        const raw = fs.readFileSync(this.configPath, "utf-8");
        const parsed = JSON.parse(raw) as PersistedPdfOcrConfig;
        if (parsed && typeof parsed === "object") {
          this.config = parsed;
        }
      }
    } catch (error) {
      console.error("Failed to load PDF OCR config:", error);
      this.config = {};
    }
  }

  private saveConfig() {
    try {
      fs.writeFileSync(this.configPath, JSON.stringify(this.config, null, 2));
    } catch (error) {
      console.error("Failed to save PDF OCR config:", error);
    }
  }

  private getPythonBinary(): string {
    const configured = this.config.pythonPath?.trim();
    if (!configured) {
      return DEFAULT_PYTHON_BIN;
    }
    const normalized = path.normalize(configured);
    if (fs.existsSync(normalized) && fs.statSync(normalized).isDirectory()) {
      const binName = process.platform === "win32" ? "python.exe" : "python";
      const candidate = path.join(normalized, process.platform === "win32" ? "Scripts" : "bin", binName);
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
    return normalized;
  }

  private applyEnvOverrides(env: NodeJS.ProcessEnv) {
    if (this.config.deepseekRepoPath) {
      env.DEEPSEEK_OCR_REPO = this.config.deepseekRepoPath;
      env.DEEPSEEK_OCR_ROOT = this.config.deepseekRepoPath;
    }
    if (this.config.huggingFaceRepoId) {
      env.DEEPSEEK_OCR_MODEL_REPO = this.config.huggingFaceRepoId;
    }
    if (this.config.huggingFaceRevision) {
      env.DEEPSEEK_OCR_MODEL_REVISION = this.config.huggingFaceRevision;
    }
  }

  private broadcastStatus() {
    this.broadcast({
      type: "status",
      payload: this.getStatus(),
    });
  }

  private handleWorkerMessage(message: PythonMessage) {
    if (message.deepseek_module_path) {
      this.lastResolvedModulePath = message.deepseek_module_path;
      this.broadcastStatus();
    }
  }

  public getConfig(): PdfOcrConfig {
    return {
      ...this.config,
      lastResolvedModulePath: this.lastResolvedModulePath,
    };
  }

  public updateConfig(updates: Partial<PersistedPdfOcrConfig>) {
    const next: PersistedPdfOcrConfig = { ...this.config };
    if (Object.prototype.hasOwnProperty.call(updates, "pythonPath")) {
      const value = updates.pythonPath;
      next.pythonPath = value && value.trim().length > 0 ? value.trim() : undefined;
    }
    if (Object.prototype.hasOwnProperty.call(updates, "deepseekRepoPath")) {
      const value = updates.deepseekRepoPath;
      next.deepseekRepoPath = value && value.trim().length > 0 ? value.trim() : undefined;
    }
    if (Object.prototype.hasOwnProperty.call(updates, "huggingFaceRepoId")) {
      const value = updates.huggingFaceRepoId;
      next.huggingFaceRepoId = value && value.trim().length > 0 ? value.trim() : undefined;
    }
    if (Object.prototype.hasOwnProperty.call(updates, "huggingFaceRevision")) {
      const value = updates.huggingFaceRevision;
      next.huggingFaceRevision = value && value.trim().length > 0 ? value.trim() : undefined;
    }
    this.config = next;
    this.saveConfig();
    this.broadcastStatus();
  }

  public async downloadModels(): Promise<void> {
    if (this.modelDownloadStatus === "in-progress") {
      throw new Error("Model download already in progress");
    }

    this.modelDownloadStatus = "in-progress";
    this.modelDownloadError = undefined;
    this.broadcastStatus();

    try {
      await this.runPython(
        ["--models-dir", this.modelsDir, "--download-models"],
        (message) => {
          if (message.event === "complete") {
            this.modelDownloadStatus = "completed";
            this.broadcastStatus();
          } else if (message.event === "error") {
            this.modelDownloadStatus = "failed";
            this.modelDownloadError = message.error || message.message;
            this.broadcastStatus();
          }
        },
        (stderrLine) => {
          console.log("PDF OCR download:", stderrLine);
        }
      );
      if (this.modelDownloadStatus === "in-progress") {
        this.modelDownloadStatus = "completed";
        this.broadcastStatus();
      }
    } catch (error) {
      this.modelDownloadStatus = "failed";
      this.modelDownloadError = error instanceof Error ? error.message : String(error);
      this.broadcastStatus();
      throw error;
    }
  }

  private toPublicJob(job: InternalJob): PdfOcrJobStatus {
    return {
      id: job.id,
      status: job.status,
      progress: Math.max(0, Math.min(100, Math.round(job.progress))),
      message: job.message,
      pageCount: job.pageCount,
      processedPages: job.processedPages,
      outputFile: job.outputFile,
      pdfFileName: job.pdfFileName,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      error: job.error,
    };
  }

  private broadcast(message: PdfOcrWsMessage) {
    this.subscribers.forEach((subscriber) => {
      try {
        subscriber(message);
      } catch (err) {
        console.error("PDF OCR subscriber error:", err);
      }
    });
  }

  private broadcastLog(jobId: string, level: "info" | "warn" | "error", message: string) {
    const payload: PdfOcrLogEntry = {
      id: nanoid(),
      jobId,
      level,
      message,
      timestamp: Date.now(),
    };
    this.broadcast({
      type: "log",
      payload,
    });
  }

  public subscribe(listener: (message: PdfOcrWsMessage) => void): () => void {
    this.subscribers.add(listener);
    listener({
      type: "status",
      payload: this.getStatus(),
    });
    return () => {
      this.subscribers.delete(listener);
    };
  }

  public getStatus(): PdfOcrStatus {
    const jobs = Array.from(this.jobs.values())
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((job) => this.toPublicJob(job));
    return {
      jobs,
      modelsDir: this.modelsDir,
      downloadStatus: this.modelDownloadStatus,
      downloadError: this.modelDownloadError,
      config: this.getConfig(),
    };
  }

  public getJob(jobId: string): PdfOcrJobStatus | undefined {
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

  private runPython(
    args: string[],
    onMessage?: (message: PythonMessage) => void,
    onError?: (message: string) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonEnv = {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      };

      this.applyEnvOverrides(pythonEnv);

      const python = spawn(this.getPythonBinary(), ["-u", this.workerScript, ...args], {
        env: pythonEnv,
      });

      python.on("error", (error) => {
        reject(error);
      });

      const rl = readline.createInterface({ input: python.stdout });
      rl.on("line", (line) => {
        try {
          const message = JSON.parse(line) as PythonMessage;
          this.handleWorkerMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.error("Failed to parse PDF OCR worker message:", line, error);
        }
      });

      const errorRl = readline.createInterface({ input: python.stderr });
      errorRl.on("line", (line) => {
        onError?.(line);
      });

      python.on("close", (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`PDF OCR worker exited with code ${code}`));
        }
      });
    });
  }

  public async startJob(params: {
    pdfBuffer: Buffer;
    fileName: string;
    totalPages: number;
  }): Promise<PdfOcrJobStatus> {
    const jobId = nanoid();
    const jobDir = path.join(this.jobsDir, jobId);
    await fsPromises.mkdir(jobDir, { recursive: true });

    const pdfPath = path.join(jobDir, "document.pdf");
    const outputPath = path.join(jobDir, "combined.txt");

    await fsPromises.writeFile(pdfPath, params.pdfBuffer);

    const createdAt = Date.now();
    const job: InternalJob = {
      id: jobId,
      status: "running",
      progress: 1,
      message: "Preparing OCR job…",
      pageCount: params.totalPages,
      processedPages: 0,
      pdfFileName: params.fileName,
      createdAt,
      updatedAt: createdAt,
      workingDir: jobDir,
    };

    this.jobs.set(jobId, job);
    this.broadcast({
      type: "job",
      payload: this.toPublicJob(job),
    });
    this.broadcastLog(jobId, "info", `Starting PDF OCR job for ${params.fileName}`);

    void this.runPython(
      [
        "--job-id",
        jobId,
        "--pdf-path",
        pdfPath,
        "--output-path",
        outputPath,
        "--models-dir",
        this.modelsDir,
        "--total-pages",
        String(Math.max(1, params.totalPages)),
      ],
      (message) => {
        if (message.event === "progress") {
          const processed = typeof message.processed_pages === "number" ? message.processed_pages : job.processedPages || 0;
          const total = typeof message.total_pages === "number" ? message.total_pages : job.pageCount || params.totalPages;
          const progress = typeof message.progress === "number"
            ? message.progress * 100
            : total > 0
              ? (processed / total) * 100
              : job.progress;
          this.updateJob(jobId, {
            processedPages: processed,
            pageCount: total,
            progress,
            message: message.message,
          });
        } else if (message.event === "log") {
          this.broadcastLog(jobId, message.level || "info", message.message || "");
        } else if (message.event === "complete") {
          const processed = typeof message.processed_pages === "number" ? message.processed_pages : job.processedPages;
          const total = typeof message.total_pages === "number" ? message.total_pages : job.pageCount;
          this.updateJob(jobId, {
            status: "completed",
            progress: 100,
            processedPages: processed,
            pageCount: total,
            message: message.message ?? "OCR complete",
            outputFile: message.output_path || outputPath,
          });
          if (message.text) {
            this.broadcast({
              type: "text",
              payload: {
                jobId,
                text: message.text,
              },
            });
          }
          this.broadcastLog(jobId, "info", message.message || "OCR job completed");
        } else if (message.event === "error") {
          const errorMessage = message.error || message.message || "Unknown OCR error";
          this.updateJob(jobId, {
            status: "failed",
            error: errorMessage,
            message: message.message,
          });
          this.broadcastLog(jobId, "error", errorMessage);
        }
      },
      (stderrLine) => {
        this.broadcastLog(jobId, "info", stderrLine);
      }
    ).catch((error) => {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.updateJob(jobId, {
        status: "failed",
        error: errorMessage,
        message: "OCR job failed",
      });
      this.broadcastLog(jobId, "error", errorMessage);
    });

    return this.toPublicJob(job);
  }
}

export const pdfOcrService = new PdfOcrService();
