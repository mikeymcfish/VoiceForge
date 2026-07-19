import fs from "fs";
import fsPromises from "fs/promises";
import path from "path";
import { spawn, type ChildProcess } from "child_process";
import readline from "readline";
import { fileURLToPath } from "url";
import { nanoid } from "nanoid";
import { Client, handle_file } from "@gradio/client";
import type {
  SpeechEngine,
  SpeechExecutionTarget,
  SpeechJobStatus,
  SpeechOutputFormat,
  SpeechReferenceEnhancement,
  SpeechStatus,
  SpeechWsMessage,
} from "@shared/schema";
import {
  MOSS_DEFAULT_TEMPERATURE,
  MOSS_DEFAULT_TOP_P,
  MOSS_DELAY_MODEL_ID,
  MOSS_LOCAL_MODEL_ID,
  mossHostedDurationTokens,
} from "@shared/moss-tts";
import { getHuggingFaceApiToken } from "./llm-service";
import { huggingFaceUsageService } from "./huggingface-usage-service";
import { endpointParameters, exactParameterNames } from "./gradio-api-contract";
import { terminateChildProcess, terminateChildProcessTree } from "./process-utils";
import {
  enhanceReferenceAudio,
  finalizeSpeechAudio,
  getAudioProcessingCapabilities,
} from "./audio-postprocess-service";

type JobState = SpeechJobStatus["status"];
type SetupState = "idle" | "in-progress" | "completed" | "failed";
type LogLevel = "info" | "warn" | "error";

type SetupProgress = {
  progress?: number;
  message?: string;
  updatedAt?: number;
  stalled?: boolean;
};

type EngineConfig = {
  engine: SpeechEngine;
  label: string;
  rootDir: string;
  modelsDir: string;
  jobsDir: string;
  workerScript: string;
  pythonEnvName: "QWEN_TTS_PYTHON" | "MOSS_TTS_PYTHON";
  defaultPython: string;
  defaultModelId: string;
  allowedModels: ReadonlySet<string>;
  spaceId: string;
  spaceHost: string;
  localModes: readonly string[];
  hostedModes: readonly string[];
};

type InternalJob = SpeechJobStatus & {
  workingDir: string;
};

type WorkerMessage = {
  event: string;
  progress?: number;
  message?: string;
  level?: LogLevel;
  output_path?: string;
  error?: string;
  model_id?: string;
};

type RemoteSubmission = {
  cancel: () => Promise<void>;
  [Symbol.asyncIterator](): AsyncIterator<any>;
};

const QWEN_BASE_06_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
const QWEN_BASE_17_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
const QWEN_MODELS = new Set([QWEN_BASE_06_MODEL, QWEN_BASE_17_MODEL]);
const MOSS_MODELS = new Set([MOSS_DELAY_MODEL_ID, MOSS_LOCAL_MODEL_ID]);
const PINNED_QWEN_REVISIONS: Record<string, string> = {
  "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "5d83992436eae1d760afd27aff78a71d676296fc",
  "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "fd4b254389122332181a7c3db7f27e918eec64e3",
};
const PINNED_MOSS_MODELS: Record<
  string,
  {
    revision: string;
    codecId: string;
    codecRevision: string;
    manifestName: string;
  }
> = {
  [MOSS_DELAY_MODEL_ID]: {
    revision: "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
    codecId: "OpenMOSS-Team/MOSS-Audio-Tokenizer",
    codecRevision: "3cd226ba2947efa357ef453bcad111b6eafba782",
    manifestName: ".voiceforge-moss-models.json",
  },
  [MOSS_LOCAL_MODEL_ID]: {
    revision: "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
    codecId: "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
    codecRevision: "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
    manifestName: ".voiceforge-moss-local-v1.5-models.json",
  },
};
const MAX_REMOTE_AUDIO_BYTES = 100 * 1024 * 1024;
const SETUP_ACTIVITY_POLL_MS = 15_000;
const SETUP_STALL_MS = 5 * 60_000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function resolveWorkerScript(name: string): string {
  const distWorker = path.join(__dirname, "python", name);
  const srcWorker = path.join(__dirname, "..", "server", "python", name);
  if (fs.existsSync(distWorker)) return distWorker;
  if (fs.existsSync(srcWorker)) return srcWorker;
  throw new Error(`Speech worker not found: ${name}`);
}

function makeEngineConfig(engine: SpeechEngine): EngineConfig {
  const baseRoot = path.join(process.cwd(), "attached_assets");
  if (engine === "qwen") {
    const rootDir = path.join(baseRoot, "qwen3-tts");
    return {
      engine,
      label: "Qwen3-TTS",
      rootDir,
      modelsDir: path.join(rootDir, "models"),
      jobsDir: path.join(rootDir, "jobs"),
      workerScript: resolveWorkerScript("qwen_tts_worker.py"),
      pythonEnvName: "QWEN_TTS_PYTHON",
      defaultPython: process.platform === "win32" ? "python" : "python3",
      defaultModelId: QWEN_BASE_06_MODEL,
      allowedModels: QWEN_MODELS,
      spaceId: "Qwen/Qwen3-TTS",
      spaceHost: "qwen-qwen3-tts.hf.space",
      localModes: ["clone"],
      hostedModes: ["clone", "design", "custom"],
    };
  }

  const rootDir = path.join(baseRoot, "moss-tts-v1.5");
  return {
    engine,
    label: "MOSS-TTS v1.5",
    rootDir,
    modelsDir: path.join(rootDir, "models"),
    jobsDir: path.join(rootDir, "jobs"),
    workerScript: resolveWorkerScript("moss_tts_worker.py"),
    pythonEnvName: "MOSS_TTS_PYTHON",
    defaultPython: process.platform === "win32" ? "python" : "python3",
    defaultModelId: MOSS_DELAY_MODEL_ID,
    allowedModels: MOSS_MODELS,
    spaceId: "OpenMOSS-Team/MOSS-TTS-v1.5",
    spaceHost: "openmoss-team-moss-tts-v1-5.hf.space",
    localModes: ["direct", "clone"],
    hostedModes: ["direct", "clone", "continuation", "continuation-clone"],
  };
}

function messageText(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const messages = value
      .map((item) =>
        item && typeof item === "object" && "message" in item
          ? String((item as { message?: unknown }).message ?? "")
          : String(item)
      )
      .filter(Boolean);
    return messages.length > 0 ? messages.join("; ") : undefined;
  }
  return value == null ? undefined : String(value);
}

function assertAllowedMode(config: EngineConfig, target: SpeechExecutionTarget, mode: string): void {
  const allowed = target === "local" ? config.localModes : config.hostedModes;
  if (!allowed.includes(mode)) {
    throw new Error(`${config.label} does not support ${mode} mode with the ${target} target.`);
  }
}

function sanitizeExtension(fileName?: string): string {
  const extension = path.extname(path.basename(fileName || "")).toLowerCase();
  return /^\.[a-z0-9]{1,8}$/.test(extension) ? extension : ".wav";
}

class SpeechOperationTimeoutError extends Error {}
class SpeechJobCancelledError extends Error {}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new SpeechOperationTimeoutError(message)), timeoutMs);
    timer.unref?.();
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

function workerEnvironment(config: EngineConfig, command: string): NodeJS.ProcessEnv {
  const exactKeys = new Set([
    "ALLUSERSPROFILE", "APPDATA", "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)",
    "COMSPEC", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "LIBRARY_PATH",
    "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "OS", "PATH", "PATHEXT", "PROCESSOR_ARCHITECTURE",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "ROCM_PATH", "SYSTEMDRIVE",
    "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "USERPROFILE", "WINDIR", "XDG_CACHE_HOME",
  ]);
  const allowedPrefixes = [
    "CUDA_", "HSA_", "KMP_", "MKL_", "MOSS_TTS_", "NVIDIA_", "OMP_", "PYTORCH_",
    "QWEN_TTS_", "ROCR_", "TORCH_", "VOICEFORGE_WORKER_",
  ];
  if (command === "setup") {
    [
      "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "REQUESTS_CA_BUNDLE",
      "SSL_CERT_DIR", "SSL_CERT_FILE",
    ].forEach((key) => exactKeys.add(key));
  }

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
  env.HF_HUB_DISABLE_TELEMETRY = "1";
  if (process.platform === "win32" && command === "setup" && config.engine === "moss") {
    // Xet can leave large Windows downloads wedged in CLOSE_WAIT. Plain Hub
    // HTTP resumes the same cache and is more reliable for this model.
    env.HF_HUB_DISABLE_XET = "1";
  }
  return env;
}

class SpeechService {
  private readonly configs: Record<SpeechEngine, EngineConfig>;
  private readonly setupStatus: Record<SpeechEngine, SetupState> = {
    qwen: "idle",
    moss: "idle",
  };
  private readonly setupErrors: Partial<Record<SpeechEngine, string>> = {};
  private readonly setupProgress: Partial<Record<SpeechEngine, SetupProgress>> = {};
  private readonly jobs = new Map<string, InternalJob>();
  private readonly subscribers = new Set<(message: SpeechWsMessage) => void>();
  private readonly activeProcesses = new Map<string, ChildProcess>();
  private readonly activeSetupProcesses = new Map<SpeechEngine, ChildProcess>();
  private readonly cancelledSetupProcesses = new Set<ChildProcess>();
  private readonly setupMonitors = new Map<SpeechEngine, NodeJS.Timeout>();
  private readonly activeRemote = new Map<string, RemoteSubmission>();
  private localQueueTail: Promise<void> = Promise.resolve();

  constructor() {
    this.configs = {
      qwen: makeEngineConfig("qwen"),
      moss: makeEngineConfig("moss"),
    };
    for (const config of Object.values(this.configs)) {
      fs.mkdirSync(config.modelsDir, { recursive: true });
      fs.mkdirSync(config.jobsDir, { recursive: true });
      if (this.listAvailableModels(config).length > 0) this.setupStatus[config.engine] = "completed";
    }
  }

  private configuredPython(config: EngineConfig): string {
    return process.env[config.pythonEnvName]?.trim() || config.defaultPython;
  }

  private isRuntimeConfigured(config: EngineConfig): boolean {
    const configured = process.env[config.pythonEnvName]?.trim();
    if (!configured) return false;
    if (!path.isAbsolute(configured)) return true;
    return fs.existsSync(configured);
  }

  private listAvailableModels(config: EngineConfig): string[] {
    if (!fs.existsSync(config.modelsDir)) return [];
    const found = new Set<string>();

    if (config.engine === "qwen") {
      try {
        const manifest = JSON.parse(
          fs.readFileSync(path.join(config.modelsDir, ".voiceforge-qwen-models.json"), "utf-8")
        ) as {
          models?: Record<string, { repo_id?: string; revision?: string; files?: unknown[] }>;
        };
        if (manifest.models && typeof manifest.models === "object") {
          for (const [key, value] of Object.entries(manifest.models)) {
            const modelId = (value.repo_id || key).trim();
            if (
              config.allowedModels.has(modelId) &&
              value.revision === PINNED_QWEN_REVISIONS[modelId] &&
              Array.isArray(value.files) &&
              value.files.length > 0
            ) {
              found.add(modelId);
            }
          }
        }
      } catch {
        // Continue with the legacy per-snapshot manifest scan below.
      }
    } else {
      for (const [modelId, pinned] of Object.entries(PINNED_MOSS_MODELS)) {
        try {
          const manifest = JSON.parse(
            fs.readFileSync(path.join(config.modelsDir, pinned.manifestName), "utf-8")
          ) as {
            artifacts?: {
              model?: { repo_id?: string; model_id?: string; revision?: string; files?: unknown[] };
              codec?: { repo_id?: string; revision?: string; files?: unknown[] };
            };
          };
          const modelArtifact = manifest.artifacts?.model;
          const codecArtifact = manifest.artifacts?.codec;
          const artifactId = (modelArtifact?.repo_id || modelArtifact?.model_id || "").trim();
          if (
            artifactId === modelId &&
            config.allowedModels.has(artifactId) &&
            modelArtifact?.revision === pinned.revision &&
            Array.isArray(modelArtifact.files) &&
            modelArtifact.files.length > 0 &&
            codecArtifact?.repo_id === pinned.codecId &&
            codecArtifact.revision === pinned.codecRevision &&
            Array.isArray(codecArtifact.files) &&
            codecArtifact.files.length > 0
          ) {
            found.add(artifactId);
          }
        } catch {
          // A missing, partial, or stale checkpoint manifest is not selectable.
        }
      }
    }

    const pending = [config.modelsDir];
    while (pending.length > 0) {
      const current = pending.pop()!;
      let entries: fs.Dirent[];
      try {
        entries = fs.readdirSync(current, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        const fullPath = path.join(current, entry.name);
        if (entry.isDirectory()) {
          pending.push(fullPath);
          continue;
        }
        if (entry.name !== "voiceforge-model.json") continue;
        try {
          const manifest = JSON.parse(fs.readFileSync(fullPath, "utf-8")) as {
            complete?: boolean;
            repo_id?: string;
            model_id?: string;
          };
          const modelId = (manifest.repo_id || manifest.model_id || "").trim();
          const trustedLegacyModel =
            config.engine !== "moss" || modelId === MOSS_DELAY_MODEL_ID;
          if (manifest.complete && trustedLegacyModel && config.allowedModels.has(modelId)) {
            found.add(modelId);
          }
        } catch {
          // An incomplete/corrupt snapshot is intentionally not selectable.
        }
      }
    }
    return [...found].sort();
  }

  private broadcast(message: SpeechWsMessage): void {
    for (const subscriber of this.subscribers) {
      try {
        subscriber(message);
      } catch (error) {
        console.error("Speech subscriber error:", error);
      }
    }
  }

  private log(level: LogLevel, message: string): void {
    this.broadcast({
      type: "log",
      payload: { id: nanoid(), level, message, timestamp: Date.now() },
    });
  }

  public subscribe(listener: (message: SpeechWsMessage) => void): () => void {
    this.subscribers.add(listener);
    listener({ type: "status", payload: this.getStatus() });
    return () => this.subscribers.delete(listener);
  }

  private publicJob(job: InternalJob): SpeechJobStatus {
    const { workingDir: _workingDir, ...result } = job;
    return result;
  }

  public getStatus(): SpeechStatus {
    const audioCapabilities = getAudioProcessingCapabilities();
    return {
      tokenConfigured: Boolean(getHuggingFaceApiToken()),
      audioProcessing: audioCapabilities,
      engines: (Object.keys(this.configs) as SpeechEngine[]).map((engine) => {
        const config = this.configs[engine];
        const availableModels = this.listAvailableModels(config);
        return {
          engine,
          setupStatus: this.setupStatus[engine],
          setupProgress: this.setupProgress[engine]?.progress,
          setupMessage: this.setupProgress[engine]?.message,
          setupUpdatedAt: this.setupProgress[engine]?.updatedAt,
          setupStalled: this.setupProgress[engine]?.stalled,
          runtimeConfigured: this.isRuntimeConfigured(config),
          modelsReady: availableModels.length > 0,
          availableModels,
          modelsPath: config.modelsDir,
          lastSetupError: this.setupErrors[engine],
          spaceId: config.spaceId,
          hostedAvailable: true,
          localModes: [...config.localModes],
          hostedModes: [...config.hostedModes],
        };
      }),
      jobs: [...this.jobs.values()]
        .sort((a, b) => b.createdAt - a.createdAt)
        .slice(0, 40)
        .map((job) => this.publicJob(job)),
    };
  }

  private updateJob(id: string, patch: Partial<InternalJob>): void {
    const current = this.jobs.get(id);
    if (!current || current.status === "cancelled") return;
    const updated: InternalJob = { ...current, ...patch, updatedAt: Date.now() };
    this.jobs.set(id, updated);
    this.broadcast({ type: "job", payload: this.publicJob(updated) });
  }

  private assertJobActive(id: string): void {
    if (this.jobs.get(id)?.status === "cancelled") {
      throw new SpeechJobCancelledError("Synthesis was cancelled.");
    }
  }

  private setupDiskFingerprint(rootDir: string): string {
    let totalBytes = 0;
    let newestMtime = 0;
    let fileCount = 0;
    const pending = [rootDir];
    while (pending.length > 0) {
      const current = pending.pop()!;
      let entries: fs.Dirent[];
      try {
        entries = fs.readdirSync(current, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        const fullPath = path.join(current, entry.name);
        if (entry.isDirectory()) {
          pending.push(fullPath);
          continue;
        }
        if (!entry.isFile()) continue;
        try {
          const stat = fs.statSync(fullPath);
          totalBytes += stat.size;
          newestMtime = Math.max(newestMtime, stat.mtimeMs);
          fileCount += 1;
        } catch {}
      }
    }
    return `${fileCount}:${totalBytes}:${Math.floor(newestMtime)}`;
  }

  private startSetupMonitor(config: EngineConfig): void {
    const previous = this.setupMonitors.get(config.engine);
    if (previous) clearInterval(previous);
    let fingerprint = this.setupDiskFingerprint(config.modelsDir);
    const timer = setInterval(() => {
      if (this.setupStatus[config.engine] !== "in-progress") return;
      const nextFingerprint = this.setupDiskFingerprint(config.modelsDir);
      const now = Date.now();
      const current = this.setupProgress[config.engine] ?? {};
      if (nextFingerprint !== fingerprint) {
        fingerprint = nextFingerprint;
        this.setupProgress[config.engine] = { ...current, updatedAt: now, stalled: false };
        this.broadcast({ type: "status", payload: this.getStatus() });
        return;
      }
      if (!current.stalled && current.updatedAt && now - current.updatedAt >= SETUP_STALL_MS) {
        this.setupProgress[config.engine] = {
          ...current,
          message: "No download activity for five minutes. The setup may be stalled; stop and retry to reuse the cache.",
          stalled: true,
        };
        this.log("warn", `${config.label} setup appears stalled; cached files will be preserved on retry.`);
        this.broadcast({ type: "status", payload: this.getStatus() });
      }
    }, SETUP_ACTIVITY_POLL_MS);
    timer.unref?.();
    this.setupMonitors.set(config.engine, timer);
  }

  private stopSetupMonitor(engine: SpeechEngine): void {
    const timer = this.setupMonitors.get(engine);
    if (timer) clearInterval(timer);
    this.setupMonitors.delete(engine);
  }

  private enqueueLocal(jobId: string, run: () => Promise<void>): Promise<void> {
    const queued = this.localQueueTail
      .catch(() => undefined)
      .then(async () => {
        this.assertJobActive(jobId);
        await run();
      });
    this.localQueueTail = queued.catch(() => undefined);
    return queued;
  }

  private runPython(
    config: EngineConfig,
    command: string,
    args: string[],
    onMessage?: (message: WorkerMessage) => void,
    jobId?: string,
    onSpawn?: (child: ChildProcess) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      let callbackError: Error | undefined;
      const child = spawn(
        this.configuredPython(config),
        [
          config.workerScript,
          "--root-dir",
          config.rootDir,
          "--models-dir",
          config.modelsDir,
          "--jobs-dir",
          config.jobsDir,
          command,
          ...args,
        ],
        {
          cwd: config.rootDir,
          detached: process.platform !== "win32",
          env: workerEnvironment(config, command),
          windowsHide: true,
        }
      );
      if (jobId) this.activeProcesses.set(jobId, child);
      if (command === "setup") this.activeSetupProcesses.set(config.engine, child);
      onSpawn?.(child);

      const release = () => {
        if (jobId && this.activeProcesses.get(jobId) === child) this.activeProcesses.delete(jobId);
        if (command === "setup" && this.activeSetupProcesses.get(config.engine) === child) {
          this.activeSetupProcesses.delete(config.engine);
        }
      };
      child.stdout.setEncoding("utf-8");
      const lines = readline.createInterface({ input: child.stdout });
      lines.on("line", (line) => {
        if (!line.trim()) return;
        let parsed: WorkerMessage;
        try {
          parsed = JSON.parse(line) as WorkerMessage;
        } catch {
          this.log("info", line);
          return;
        }
        try {
          onMessage?.(parsed);
        } catch (error) {
          callbackError = error instanceof Error ? error : new Error(String(error));
          terminateChildProcess(child, { processGroup: process.platform !== "win32" });
        }
      });
      child.stderr.setEncoding("utf-8");
      child.stderr.on("data", (chunk: string) => {
        chunk
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .forEach((line) => this.log("warn", line));
      });
      child.once("error", (error) => {
        release();
        lines.close();
        reject(error);
      });
      child.once("exit", release);
      child.once("close", (code) => {
        release();
        lines.close();
        if (callbackError) reject(callbackError);
        else if (code === 0) resolve();
        else reject(new Error(`${config.label} worker exited with code ${code}.`));
      });
    });
  }

  public validateSetupRequest(engine: SpeechEngine, modelId?: string): string {
    const config = this.configs[engine];
    const selectedModel = modelId?.trim() || config.defaultModelId;
    if (!config.allowedModels.has(selectedModel)) throw new Error("Unsupported model selection.");
    if (!this.isRuntimeConfigured(config)) {
      throw new Error(
        `${config.pythonEnvName} is not configured. Run VoiceForge.cmd setup-${engine} and restart VoiceForge.`
      );
    }
    if (this.setupStatus[engine] === "in-progress") throw new Error("Setup is already in progress.");
    return selectedModel;
  }

  public async startSetup(engine: SpeechEngine, modelId?: string): Promise<void> {
    const config = this.configs[engine];
    const selectedModel = this.validateSetupRequest(engine, modelId);

    this.setupStatus[engine] = "in-progress";
    delete this.setupErrors[engine];
    this.setupProgress[engine] = {
      progress: 1,
      message: "Starting the pinned model download...",
      updatedAt: Date.now(),
      stalled: false,
    };
    this.startSetupMonitor(config);
    this.broadcast({ type: "status", payload: this.getStatus() });
    let setupChild: ChildProcess | undefined;
    try {
      const setupArgs = ["--model-id", selectedModel];
      await this.runPython(
        config,
        "setup",
        setupArgs,
        (message) => {
          if (message.event === "log" || message.event === "progress") {
            this.log(message.level || "info", message.message || `${config.label} setup is running.`);
          }
          if (message.event === "progress") {
            const raw = typeof message.progress === "number" ? message.progress : undefined;
            this.setupProgress[engine] = {
              progress:
                raw === undefined
                  ? this.setupProgress[engine]?.progress
                  : Math.max(0, Math.min(100, raw <= 1 ? raw * 100 : raw)),
              message: message.message || `${config.label} setup is running.`,
              updatedAt: Date.now(),
              stalled: false,
            };
            this.broadcast({ type: "status", payload: this.getStatus() });
          }
          if (message.event === "error") throw new Error(message.error || message.message || "Setup failed.");
        },
        undefined,
        (child) => {
          setupChild = child;
        }
      );
      if (setupChild) this.cancelledSetupProcesses.delete(setupChild);
      this.setupStatus[engine] = "completed";
      this.setupProgress[engine] = {
        progress: 100,
        message: `${config.label} pinned model setup completed.`,
        updatedAt: Date.now(),
        stalled: false,
      };
      this.log("info", `${config.label} pinned model setup completed.`);
    } catch (error) {
      if (setupChild && this.cancelledSetupProcesses.delete(setupChild)) {
        this.setupStatus[engine] = "idle";
        this.setupErrors[engine] = "Setup stopped. Cached model files were preserved for the next retry.";
        this.setupProgress[engine] = {
          ...this.setupProgress[engine],
          message: this.setupErrors[engine],
          updatedAt: Date.now(),
          stalled: false,
        };
        this.log("info", `${config.label} setup stopped; cached files were preserved.`);
        return;
      }
      this.setupStatus[engine] = "failed";
      this.setupErrors[engine] = error instanceof Error ? error.message : String(error);
      this.setupProgress[engine] = {
        ...this.setupProgress[engine],
        message: this.setupErrors[engine],
        updatedAt: Date.now(),
        stalled: false,
      };
      this.log("error", `${config.label} setup failed: ${this.setupErrors[engine]}`);
      throw error;
    } finally {
      if (setupChild) this.cancelledSetupProcesses.delete(setupChild);
      this.stopSetupMonitor(engine);
      this.broadcast({ type: "status", payload: this.getStatus() });
    }
  }

  public async cancelSetup(engine: SpeechEngine): Promise<void> {
    if (this.setupStatus[engine] !== "in-progress") throw new Error("Setup is not in progress.");
    const child = this.activeSetupProcesses.get(engine);
    if (!child) throw new Error("The setup worker is not available to stop.");
    this.cancelledSetupProcesses.add(child);
    this.setupProgress[engine] = {
      ...this.setupProgress[engine],
      message: "Stopping setup and preserving cached files...",
      updatedAt: Date.now(),
      stalled: false,
    };
    this.broadcast({ type: "status", payload: this.getStatus() });

    const closed = new Promise<void>((resolve) => child.once("close", () => resolve()));
    let terminationRequested: boolean;
    try {
      terminationRequested = await withTimeout(
        terminateChildProcessTree(child, { processGroup: process.platform !== "win32" }),
        10_000,
        "Timed out while requesting setup worker termination."
      );
    } catch (error) {
      this.cancelledSetupProcesses.delete(child);
      throw error;
    }
    if (!terminationRequested) {
      this.cancelledSetupProcesses.delete(child);
      throw new Error("The setup worker could not be stopped.");
    }
    await withTimeout(closed, 15_000, "Timed out while stopping the setup worker.");
    // Let the rejected setup task run its catch/finally handlers before a
    // caller starts a replacement worker for the same engine.
    await new Promise<void>((resolve) => setImmediate(resolve));
    if (this.setupStatus[engine] === "in-progress") {
      this.setupStatus[engine] = "idle";
      this.setupErrors[engine] = "Setup stopped. Cached model files were preserved for the next retry.";
      this.setupProgress[engine] = {
        ...this.setupProgress[engine],
        message: this.setupErrors[engine],
        updatedAt: Date.now(),
        stalled: false,
      };
      this.stopSetupMonitor(engine);
      this.broadcast({ type: "status", payload: this.getStatus() });
    }
  }

  public async startSynthesis(params: {
    engine: SpeechEngine;
    target: SpeechExecutionTarget;
    mode: string;
    text: string;
    voiceBuffer?: Buffer;
    voiceFileName?: string;
    modelId?: string;
    referenceText?: string;
    language?: string;
    xVectorOnly?: boolean;
    voiceDescription?: string;
    speaker?: string;
    instruction?: string;
    modelSize?: string;
    durationControl?: boolean;
    durationTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
    maxNewTokens?: number;
    maxChars?: number;
    gapMs?: number;
    outputFormat?: SpeechOutputFormat;
    useChapters?: boolean;
    chapterPauseMs?: number;
    mp3Quality?: number;
    normalizeLevels?: boolean;
    referenceEnhancement?: SpeechReferenceEnhancement;
    audioSrModel?: "speech" | "basic";
    audioSrDevice?: string;
    audioSrDdimSteps?: number;
    audioSrGuidanceScale?: number;
    audioSrSeed?: number;
  }): Promise<SpeechJobStatus> {
    const config = this.configs[params.engine];
    assertAllowedMode(config, params.target, params.mode);
    let modelId = params.modelId?.trim() || config.defaultModelId;
    if (params.engine === "qwen" && params.target === "hf-space") {
      if (params.mode === "design") modelId = QWEN_BASE_17_MODEL;
      else if (params.modelSize === "0.6B") modelId = QWEN_BASE_06_MODEL;
      else if (params.modelSize === "1.7B") modelId = QWEN_BASE_17_MODEL;
    }
    if (!config.allowedModels.has(modelId)) throw new Error("Unsupported model selection.");
    if (
      params.engine === "moss" &&
      params.target === "hf-space" &&
      modelId !== MOSS_DELAY_MODEL_ID
    ) {
      throw new Error("The official MOSS ZeroGPU Space serves only MOSS-TTS v1.5 8B.");
    }
    if (params.target === "local") {
      if (!this.isRuntimeConfigured(config)) {
        throw new Error(`Run VoiceForge.cmd setup-${params.engine}, restart, and download the model first.`);
      }
      if (!this.listAvailableModels(config).includes(modelId)) {
        throw new Error(`Download the pinned ${modelId} snapshot before local synthesis.`);
      }
    } else if (!getHuggingFaceApiToken()) {
      throw new Error("Add a Hugging Face token before using the hosted Space.");
    }

    const useChapters = params.useChapters === true;
    const outputFormat: SpeechOutputFormat = useChapters ? "mp3" : params.outputFormat ?? "wav";
    const normalizeLevels = params.normalizeLevels ?? true;
    const referenceEnhancement: SpeechReferenceEnhancement =
      params.referenceEnhancement ?? "none";
    const audioCapabilities = getAudioProcessingCapabilities();
    if (useChapters && params.target !== "local") {
      throw new Error("Exact MP3 chapter timing is available for Local synthesis only.");
    }
    const chapterMarkerCount = (params.text.match(/\[chapter\]/gi) || []).length;
    const hasChapterContent = params.text
      .split(/\[chapter\]/i)
      .slice(1)
      .some((section) => section.trim().length > 0);
    if (useChapters && (chapterMarkerCount === 0 || !hasChapterContent)) {
      throw new Error(
        "Add at least one [CHAPTER] marker followed by spoken text before enabling MP3 chapters."
      );
    }
    if (useChapters && chapterMarkerCount > 500) {
      throw new Error("A synthesis job may contain at most 500 [CHAPTER] markers.");
    }
    if (referenceEnhancement !== "none" && !params.voiceBuffer) {
      throw new Error("Reference-audio enhancement requires a voice reference.");
    }
    if (
      (outputFormat === "mp3" || normalizeLevels || referenceEnhancement === "cleanup") &&
      !audioCapabilities.ffmpegAvailable
    ) {
      throw new Error("FFmpeg is required for the requested audio processing.");
    }
    if (referenceEnhancement === "audiosr" && !audioCapabilities.audioSrAvailable) {
      throw new Error(
        "AudioSR is unavailable. Configure VOICEFORGE_AUDIOSR_BIN to an isolated AudioSR executable and restart VoiceForge."
      );
    }

    const jobId = nanoid();
    const workingDir = path.join(config.jobsDir, jobId);
    await fsPromises.mkdir(workingDir, { recursive: true });
    const textPath = path.join(workingDir, "script.txt");
    const rawOutputPath = path.join(workingDir, "output.wav");
    const chapterManifestPath = path.join(workingDir, "chapters.json");
    await fsPromises.writeFile(textPath, params.text, "utf-8");
    let voicePath: string | undefined;
    if (params.voiceBuffer) {
      voicePath = path.join(workingDir, `reference${sanitizeExtension(params.voiceFileName)}`);
      await fsPromises.writeFile(voicePath, params.voiceBuffer);
    }

    const now = Date.now();
    const job: InternalJob = {
      id: jobId,
      engine: params.engine,
      target: params.target,
      mode: params.mode,
      status: "queued",
      progress: 2,
      message: params.target === "local" ? "Waiting for the local GPU…" : "Connecting to Hugging Face Space…",
      voiceFileName: params.voiceFileName,
      modelId,
      outputFormat,
      outputMimeType: outputFormat === "mp3" ? "audio/mpeg" : "audio/wav",
      referenceEnhancement,
      levelNormalized: normalizeLevels,
      createdAt: now,
      updatedAt: now,
      workingDir,
    };
    this.jobs.set(jobId, job);
    this.broadcast({ type: "job", payload: this.publicJob(job) });

    const effectiveModelSize =
      params.engine === "qwen" && params.target === "hf-space"
        ? params.mode === "design"
          ? "1.7B"
          : params.modelSize ?? (modelId.includes("0.6B") ? "0.6B" : "1.7B")
        : params.modelSize;
    const effectiveParams = {
      ...params,
      text: params.text,
      modelId,
      modelSize: effectiveModelSize,
      outputFormat,
      useChapters,
      normalizeLevels,
      referenceEnhancement,
    };
    const execute = async () => {
      const effectiveVoicePath = await this.prepareReferenceAudio(
        job,
        effectiveParams,
        voicePath
      );
      if (params.target === "local") {
        await this.runLocal(
          job,
          config,
          effectiveParams,
          textPath,
          rawOutputPath,
          chapterManifestPath,
          effectiveVoicePath
        );
      } else {
        await this.runHosted(
          job,
          config,
          effectiveParams,
          rawOutputPath,
          effectiveVoicePath
        );
      }
    };
    const runner =
      params.target === "local" || referenceEnhancement === "audiosr"
        ? this.enqueueLocal(job.id, execute)
        : execute();
    void runner.catch((error) => {
      if (this.jobs.get(jobId)?.status === "cancelled") return;
      const detail = error instanceof Error ? error.message : String(error);
      huggingFaceUsageService.noteZeroGpuError(detail);
      this.updateJob(jobId, { status: "failed", progress: 100, message: "Synthesis failed", error: detail });
      this.log("error", `${config.label} job ${jobId} failed: ${detail}`);
    });

    return this.publicJob(job);
  }

  private async withManagedAudioProcess<T>(
    jobId: string,
    operation: (onSpawn: (child: ChildProcess) => void) => Promise<T>
  ): Promise<T> {
    let activeChild: ChildProcess | undefined;
    try {
      return await operation((child) => {
        activeChild = child;
        this.activeProcesses.set(jobId, child);
      });
    } finally {
      if (activeChild && this.activeProcesses.get(jobId) === activeChild) {
        this.activeProcesses.delete(jobId);
      }
    }
  }

  private async prepareReferenceAudio(
    job: InternalJob,
    params: Parameters<SpeechService["startSynthesis"]>[0],
    voicePath?: string
  ): Promise<string | undefined> {
    const mode = params.referenceEnhancement ?? "none";
    if (!voicePath || mode === "none") return voicePath;
    this.assertJobActive(job.id);
    this.updateJob(job.id, {
      status: "running",
      progress: 3,
      message:
        mode === "audiosr"
          ? "Enhancing the reference with AudioSR…"
          : "Cleaning the reference audio for cloning…",
    });
    const enhancedPath = await this.withManagedAudioProcess(job.id, (onSpawn) =>
      enhanceReferenceAudio({
        inputPath: voicePath,
        workingDir: job.workingDir,
        mode,
        audioSrModel: params.audioSrModel ?? "speech",
        audioSrDevice: params.audioSrDevice ?? "auto",
        audioSrDdimSteps: params.audioSrDdimSteps ?? 50,
        audioSrGuidanceScale: params.audioSrGuidanceScale ?? 3.5,
        audioSrSeed: params.audioSrSeed ?? 42,
        onSpawn,
        assertActive: () => this.assertJobActive(job.id),
        onLog: (message) => this.log("info", message),
      })
    );
    this.assertJobActive(job.id);
    this.log(
      "info",
      `Speech job ${job.id} prepared its reference with ${
        mode === "audiosr" ? "AudioSR" : "gentle FFmpeg cleanup"
      }.`
    );
    return enhancedPath;
  }

  private async finalizeOutput(
    job: InternalJob,
    params: Parameters<SpeechService["startSynthesis"]>[0],
    rawOutputPath: string,
    chapterManifestPath?: string
  ): Promise<{
    outputPath: string;
    format: SpeechOutputFormat;
    mimeType: "audio/wav" | "audio/mpeg";
    chapterCount: number;
  }> {
    this.assertJobActive(job.id);
    const outputFormat = params.outputFormat ?? "wav";
    const normalizeLevels = params.normalizeLevels ?? true;
    if (outputFormat === "mp3" || normalizeLevels) {
      this.updateJob(job.id, {
        status: "running",
        progress: 97,
        message:
          outputFormat === "mp3"
            ? params.useChapters
              ? normalizeLevels
                ? "Normalizing levels and encoding chaptered MP3…"
                : "Encoding chaptered MP3…"
              : normalizeLevels
                ? "Normalizing levels and encoding MP3…"
                : "Encoding MP3…"
            : "Normalizing output level…",
      });
    }
    return this.withManagedAudioProcess(job.id, (onSpawn) =>
      finalizeSpeechAudio({
        inputWavPath: rawOutputPath,
        workingDir: job.workingDir,
        outputFormat,
        normalizeLevels,
        chapterManifestPath: params.useChapters ? chapterManifestPath : undefined,
        mp3Quality: params.mp3Quality ?? 2,
        onSpawn,
        assertActive: () => this.assertJobActive(job.id),
        onLog: (message) => this.log("info", message),
      })
    );
  }

  private async runLocal(
    job: InternalJob,
    config: EngineConfig,
    params: Parameters<SpeechService["startSynthesis"]>[0],
    textPath: string,
    outputPath: string,
    chapterManifestPath: string,
    voicePath?: string
  ): Promise<void> {
    this.assertJobActive(job.id);
    const args = ["--text", textPath, "--output", outputPath];
    args.unshift("--model-id", job.modelId!);
    if (voicePath && !(params.engine === "moss" && params.mode === "direct")) {
      args.push("--voice", voicePath);
    }
    if (params.language) args.push("--language", params.language);
    if (params.engine === "qwen") {
      args.push("--reference-text", params.xVectorOnly ? "" : params.referenceText || "");
      args.push("--max-chars", String(params.maxChars ?? 320));
      args.push("--gap-ms", String(params.gapMs ?? 120));
    } else {
      args.push("--temperature", String(params.temperature ?? MOSS_DEFAULT_TEMPERATURE));
      args.push("--top-p", String(params.topP ?? MOSS_DEFAULT_TOP_P));
      args.push("--top-k", String(params.topK ?? 25));
      args.push("--repetition-penalty", String(params.repetitionPenalty ?? 1));
      args.push("--max-new-tokens", String(params.maxNewTokens ?? 4096));
      args.push("--max-chars", String(params.maxChars ?? 1800));
      args.push("--gap-ms", String(params.gapMs ?? 120));
    }

    this.updateJob(job.id, { status: "running", progress: 5, message: "Running locally…" });
    args.push(
      "--chapter-pause-ms",
      String(params.useChapters ? params.chapterPauseMs ?? 0 : 0)
    );
    if (params.useChapters) args.push("--chapter-manifest", chapterManifestPath);
    await this.runPython(
      config,
      "synthesize",
      args,
      (message) => {
        if (message.event === "progress") {
          const raw = typeof message.progress === "number" ? message.progress : 0;
          this.updateJob(job.id, {
            progress: raw <= 1 ? Math.max(5, raw * 90) : Math.min(95, raw),
            message: message.message,
          });
        } else if (message.event === "log") {
          this.log(message.level || "info", message.message || "");
        } else if (message.event === "error") {
          throw new Error(message.error || message.message || "Local synthesis failed.");
        }
      },
      job.id
    );
    if (!fs.existsSync(outputPath)) throw new Error("The local worker exited without creating audio.");
    const finalized = await this.finalizeOutput(
      job,
      params,
      outputPath,
      chapterManifestPath
    );
    this.updateJob(job.id, {
      status: "completed",
      progress: 100,
      message: "Local synthesis complete",
      outputFile: finalized.outputPath,
      outputFormat: finalized.format,
      outputMimeType: finalized.mimeType,
      chapterCount: finalized.chapterCount,
    });
  }

  private expectedEndpoint(config: EngineConfig, mode: string): { name: string; parameters: string[] } {
    if (config.engine === "qwen") {
      if (mode === "design") {
        return { name: "/generate_voice_design", parameters: ["text", "language", "voice_description"] };
      }
      if (mode === "custom") {
        return {
          name: "/generate_custom_voice",
          parameters: ["text", "language", "speaker", "instruct", "model_size"],
        };
      }
      return {
        name: "/generate_voice_clone",
        parameters: ["ref_audio", "ref_text", "target_text", "language", "use_xvector_only", "model_size"],
      };
    }
    return {
      name: "/run_inference",
      parameters: [
        "text",
        "reference_audio",
        "mode_with_reference",
        "duration_control_enabled",
        "duration_tokens",
        "language_tag",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
      ],
    };
  }

  private hostedPayload(
    config: EngineConfig,
    params: Parameters<SpeechService["startSynthesis"]>[0],
    voicePath?: string
  ): Record<string, unknown> {
    if (config.engine === "qwen") {
      if (params.mode === "design") {
        return {
          text: params.text,
          language: params.language || "Auto",
          voice_description: params.voiceDescription || "Natural, clear narration",
        };
      }
      if (params.mode === "custom") {
        return {
          text: params.text,
          language: params.language || "Auto",
          speaker: params.speaker || "Ryan",
          instruct: params.instruction || "",
          model_size: params.modelSize || (params.modelId?.includes("0.6B") ? "0.6B" : "1.7B"),
        };
      }
      if (!voicePath) throw new Error("Qwen voice cloning requires a reference audio file.");
      return {
        ref_audio: handle_file(voicePath),
        ref_text: params.referenceText || "",
        target_text: params.text,
        language: params.language || "Auto",
        use_xvector_only: Boolean(params.xVectorOnly || !params.referenceText?.trim()),
        model_size: params.modelSize || (params.modelId?.includes("0.6B") ? "0.6B" : "1.7B"),
      };
    }

    const modeMap: Record<string, string> = {
      direct: "Clone",
      clone: "Clone",
      continuation: "Continuation",
      "continuation-clone": "Continuation + Clone",
    };
    if (params.mode !== "direct" && !voicePath) {
      throw new Error("This MOSS mode requires a reference audio file.");
    }
    const effectiveVoicePath = params.mode === "direct" ? undefined : voicePath;
    return {
      text: params.text,
      reference_audio: effectiveVoicePath ? handle_file(effectiveVoicePath) : null,
      mode_with_reference: modeMap[params.mode] || "Clone",
      duration_control_enabled: Boolean(params.durationControl),
      duration_tokens: mossHostedDurationTokens(
        Boolean(params.durationControl),
        params.durationTokens
      ),
      language_tag: params.language || "Auto (omit)",
      temperature: params.temperature ?? MOSS_DEFAULT_TEMPERATURE,
      top_p: params.topP ?? MOSS_DEFAULT_TOP_P,
      top_k: params.topK ?? 25,
      repetition_penalty: params.repetitionPenalty ?? 1,
      max_new_tokens: params.maxNewTokens ?? 4096,
    };
  }

  private async runHosted(
    job: InternalJob,
    config: EngineConfig,
    params: Parameters<SpeechService["startSynthesis"]>[0],
    outputPath: string,
    voicePath?: string
  ): Promise<void> {
    const token = getHuggingFaceApiToken();
    if (!token) throw new Error("A Hugging Face token is required.");
    if (!token.startsWith("hf_")) throw new Error("The configured Hugging Face token is invalid.");

    const connectPromise = Client.connect(config.spaceId, {
      token: token as `hf_${string}`,
      events: ["data", "status"],
    });
    let client: Awaited<ReturnType<typeof Client.connect>>;
    try {
      client = await withTimeout(
        connectPromise,
        30_000,
        `Timed out connecting to ${config.spaceId}.`
      );
    } catch (error) {
      if (error instanceof SpeechOperationTimeoutError) {
        void connectPromise.then((lateClient) => lateClient.close()).catch(() => undefined);
      }
      throw error;
    }
    try {
      this.assertJobActive(job.id);
      const expected = this.expectedEndpoint(config, params.mode);
      const api = await withTimeout(
        client.view_api(),
        30_000,
        `Timed out reading the ${config.spaceId} API contract.`
      );
      this.assertJobActive(job.id);
      let actualParameters: string[] | undefined;
      try {
        actualParameters = endpointParameters(api, expected.name);
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(`${config.spaceId} returned unsafe API metadata: ${detail}`);
      }
      if (!exactParameterNames(actualParameters, expected.parameters)) {
        throw new Error(
          `${config.spaceId} API changed. Expected ${expected.name}(${expected.parameters.join(", ")}); ` +
          `received ${actualParameters ? `${expected.name}(${actualParameters.join(", ")})` : "no matching endpoint"}; ` +
          "refusing an unsafe call."
        );
      }

      this.assertJobActive(job.id);
      const submission = client.submit(
        expected.name,
        this.hostedPayload(config, params, voicePath)
      ) as RemoteSubmission;
      this.activeRemote.set(job.id, submission);
      this.updateJob(job.id, { status: "queued", progress: 8, message: "Queued on Hugging Face ZeroGPU…" });

      let resultData: unknown[] | undefined;
      for await (const event of submission) {
        this.assertJobActive(job.id);
        if (event?.type === "status") {
          const stage = String(event.stage || "pending");
          const queuePosition = Number.isFinite(event.position) ? Math.max(0, Number(event.position)) : undefined;
          const etaSeconds = Number.isFinite(event.eta) ? Math.max(0, Number(event.eta)) : undefined;
          if (stage === "error") {
            throw new Error(messageText(event.message) || event.original_msg || "Hosted Space request failed.");
          }
          this.updateJob(job.id, {
            status: stage === "generating" || stage === "streaming" ? "running" : "queued",
            progress: stage === "generating" || stage === "streaming" ? 55 : 15,
            queuePosition,
            etaSeconds,
            message:
              stage === "generating" || stage === "streaming"
                ? "Generating on Hugging Face ZeroGPU…"
                : queuePosition !== undefined
                  ? `Waiting in ZeroGPU queue (position ${queuePosition})…`
                  : "Waiting in ZeroGPU queue…",
          });
        } else if (event?.type === "data" && Array.isArray(event.data)) {
          resultData = event.data;
          this.updateJob(job.id, { status: "running", progress: 90, message: "Copying hosted audio locally…" });
        }
      }

      if (!resultData || resultData.length === 0) throw new Error("The hosted Space returned no audio result.");
      this.assertJobActive(job.id);
      const statusMessage = messageText(resultData[1]);
      if (statusMessage && /\berror\b/i.test(statusMessage)) throw new Error(statusMessage);
      await this.downloadHostedAudio(config, resultData[0], outputPath, token, job.id);
      this.assertJobActive(job.id);
      const finalized = await this.finalizeOutput(job, params, outputPath);
      this.assertJobActive(job.id);
      huggingFaceUsageService.noteZeroGpuSuccess();
      this.updateJob(job.id, {
        status: "completed",
        progress: 100,
        message: statusMessage || "Hosted synthesis complete",
        outputFile: finalized.outputPath,
        outputFormat: finalized.format,
        outputMimeType: finalized.mimeType,
        chapterCount: finalized.chapterCount,
        queuePosition: undefined,
        etaSeconds: undefined,
      });
    } finally {
      this.activeRemote.delete(job.id);
      try {
        await client.close();
      } catch {
        // Closing a failed/cancelled Gradio session must not mask the job result.
      }
    }
  }

  private async downloadHostedAudio(
    config: EngineConfig,
    audio: unknown,
    outputPath: string,
    token: string,
    jobId: string
  ): Promise<void> {
    let rawUrl: string | undefined;
    if (typeof audio === "string") rawUrl = audio;
    else if (audio && typeof audio === "object") {
      const value = audio as { url?: unknown; path?: unknown };
      if (typeof value.url === "string") rawUrl = value.url;
      else if (typeof value.path === "string" && /^https?:\/\//i.test(value.path)) rawUrl = value.path;
    }
    if (!rawUrl) throw new Error("The hosted Space returned an unsupported audio reference.");

    const url = new URL(rawUrl, `https://${config.spaceHost}/`);
    if (url.protocol !== "https:" || url.hostname !== config.spaceHost) {
      throw new Error("The hosted Space returned audio from an unexpected host.");
    }
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
      redirect: "follow",
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) throw new Error(`Failed to retrieve hosted audio (${response.status}).`);
    const finalUrl = new URL(response.url);
    if (finalUrl.protocol !== "https:" || finalUrl.hostname !== config.spaceHost) {
      throw new Error("Hosted audio retrieval redirected to an unexpected host.");
    }
    const declaredSize = Number(response.headers.get("content-length") || 0);
    if (declaredSize > MAX_REMOTE_AUDIO_BYTES) throw new Error("Hosted audio exceeds the 100 MB safety limit.");
    if (!response.body) throw new Error("The hosted Space returned no audio body.");

    const temporaryPath = `${outputPath}.${nanoid(8)}.part`;
    const reader = response.body.getReader();
    let fileHandle: Awaited<ReturnType<typeof fsPromises.open>> | undefined;
    let completed = false;
    let received = 0;
    try {
      fileHandle = await fsPromises.open(temporaryPath, "wx", 0o600);
      while (true) {
        this.assertJobActive(jobId);
        const chunk = await reader.read();
        if (chunk.done) break;
        received += chunk.value.byteLength;
        if (received > MAX_REMOTE_AUDIO_BYTES) {
          throw new Error("Hosted audio exceeds the 100 MB safety limit.");
        }
        let offset = 0;
        while (offset < chunk.value.byteLength) {
          const { bytesWritten } = await fileHandle.write(
            chunk.value,
            offset,
            chunk.value.byteLength - offset,
            null
          );
          if (bytesWritten <= 0) throw new Error("Failed to write hosted audio safely.");
          offset += bytesWritten;
        }
      }
      if (received === 0) throw new Error("The hosted Space returned an empty audio file.");
      await fileHandle.sync();
      await fileHandle.close();
      fileHandle = undefined;
      await fsPromises.rename(temporaryPath, outputPath);
      completed = true;
    } finally {
      await reader.cancel().catch(() => undefined);
      if (fileHandle) await fileHandle.close().catch(() => undefined);
      if (!completed) await fsPromises.rm(temporaryPath, { force: true }).catch(() => undefined);
    }
  }

  public getJob(id: string): SpeechJobStatus | undefined {
    const job = this.jobs.get(id);
    return job ? this.publicJob(job) : undefined;
  }

  public getJobOutputPath(id: string): string | undefined {
    const job = this.jobs.get(id);
    if (!job?.outputFile || job.status !== "completed") return undefined;
    const resolved = path.resolve(job.outputFile);
    const root = path.resolve(job.workingDir);
    if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) return undefined;
    return resolved;
  }

  public cancelJob(id: string): SpeechJobStatus | undefined {
    const job = this.jobs.get(id);
    if (!job) return undefined;
    if (job.status !== "queued" && job.status !== "running") return this.publicJob(job);
    this.updateJob(id, {
      status: "cancelled",
      progress: 100,
      message: "Synthesis cancelled",
      error: undefined,
      outputFile: undefined,
      queuePosition: undefined,
      etaSeconds: undefined,
    });
    const child = this.activeProcesses.get(id);
    if (child) terminateChildProcess(child, { processGroup: process.platform !== "win32" });
    const submission = this.activeRemote.get(id);
    if (submission) void submission.cancel().catch(() => undefined);
    this.log("info", `Speech job ${id} cancellation requested.`);
    return this.getJob(id);
  }
}

export const speechService = new SpeechService();
