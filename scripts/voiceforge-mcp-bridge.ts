import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import fs from "node:fs";
import fsPromises from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const BRIDGE_VERSION = "0.1.0";
const REMOTE_MAJOR_VERSION = 2;
const REQUIRED_REMOTE_TOOLS = new Set([
  "voiceforge_list_models",
  "voiceforge_list_voices",
  "voiceforge_recommend_model",
  "voiceforge_generate_speech",
  "voiceforge_get_job",
  "voiceforge_cancel_job",
]);
const MUTATING_REMOTE_TOOLS = new Set(["voiceforge_generate_speech", "voiceforge_cancel_job"]);
const STARTUP_TIMEOUT_MS = 5 * 60_000;
const REMOTE_OPERATION_TIMEOUT_MS = 30 * 60_000;
const DISCOVERY_TIMEOUT_MS = 2_500;
const STARTUP_LOCK_STALE_MS = 10 * 60_000;
const MAX_LOCAL_AUDIO_BYTES = 250 * 1024 * 1024;
const MODEL_IDS = [
  "index-tts-2",
  "vibevoice-1.5b",
  "vibevoice-large",
  "qwen3-tts-0.6b",
  "qwen3-tts-1.7b",
  "moss-tts-v1.5",
] as const;
const MODES = ["auto", "direct", "clone", "design", "preset", "continuation", "continuation-clone"] as const;

const emptyObjectSchema = { type: "object", properties: {}, additionalProperties: false } as const;

export const VOICEFORGE_TOOLS = [
  {
    name: "voiceforge_list_models",
    title: "List VoiceForge speech models",
    description: "List VoiceForge TTS models, supported Local/Agent modes, text limits, and readiness. VoiceForge starts automatically if needed.",
    inputSchema: emptyObjectSchema,
    annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
  },
  {
    name: "voiceforge_list_voices",
    title: "List default VoiceForge voices",
    description: "List safe IDs and transcripts for VoiceForge reference voices. Filesystem source paths are not returned.",
    inputSchema: emptyObjectSchema,
    annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
  },
  {
    name: "voiceforge_recommend_model",
    title: "Recommend a VoiceForge speech model",
    description: "Recommend a model from text length, Local/Agent preference, voice availability, speakers, mode, and speed/quality preference.",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string" },
        character_count: { type: "integer", minimum: 0 },
        target: { type: "string", enum: ["auto", "local", "agent"], default: "auto" },
        has_voice: { type: "boolean", default: false },
        speaker_count: { type: "integer", minimum: 1, maximum: 5, default: 1 },
        mode: { type: "string", enum: [...MODES], default: "auto" },
        preference: { type: "string", enum: ["speed", "quality"], default: "quality" },
        needs_inline_pauses: { type: "boolean", default: false },
      },
      anyOf: [{ required: ["text"] }, { required: ["character_count"] }],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
  },
  {
    name: "voiceforge_generate_speech",
    title: "Generate speech with VoiceForge",
    description: "Start asynchronous TTS. Local stays on this machine; Agent uploads text/reference audio to the official Hugging Face Space and consumes quota. Level normalization, MP3 export, exact chapters, and reference-audio enhancement are Qwen/MOSS-only; exact chapters require Local, MP3, and [CHAPTER] markers.",
    inputSchema: {
      type: "object",
      properties: {
        request_id: { type: "string", pattern: "^[A-Za-z0-9_-]{8,128}$" },
        text: { type: "string", minLength: 1 },
        model: { type: "string", enum: ["auto", ...MODEL_IDS], default: "auto" },
        target: { type: "string", enum: ["local", "agent"] },
        mode: { type: "string", enum: [...MODES], default: "auto" },
        voice_id: { type: "string" },
        voice_ids: { type: "array", items: { type: "string" }, maxItems: 4 },
        language: { type: "string", maxLength: 40 },
        reference_text: { type: "string", maxLength: 20_000 },
        voice_description: { type: "string", maxLength: 1_000 },
        speaker: { type: "string", enum: ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"] },
        instruction: { type: "string", maxLength: 1_000 },
        preference: { type: "string", enum: ["speed", "quality"], default: "quality" },
        style: { type: "string", maxLength: 500 },
        guidance_scale: { type: "number", minimum: 0.5, maximum: 3 },
        needs_inline_pauses: { type: "boolean", default: false },
        output_format: {
          type: "string",
          enum: ["wav", "mp3"],
          description: "Qwen/MOSS only. Final managed audio format; defaults to wav.",
        },
        use_chapters: {
          type: "boolean",
          description: "Qwen/MOSS only. Exact MP3 chapters; requires Local, output_format=mp3, and [CHAPTER] markers. Defaults to false.",
        },
        chapter_pause_ms: {
          type: "integer",
          minimum: 0,
          maximum: 10_000,
          description: "Qwen/MOSS only. Extra silence before each later chapter's spoken audio in milliseconds; defaults to 0.",
        },
        mp3_quality: {
          type: "integer",
          minimum: 0,
          maximum: 9,
          description: "Qwen/MOSS only. FFmpeg VBR quality for non-chaptered MP3 (0 best, 9 smallest); defaults to 2.",
        },
        normalize_levels: {
          type: "boolean",
          description: "Qwen/MOSS only. Normalize the assembled output to a consistent speech loudness target; defaults to true.",
        },
        reference_enhancement: {
          type: "string",
          enum: ["none", "cleanup", "audiosr"],
          description: "Qwen/MOSS only. Reference voice enhancement; defaults to none.",
        },
        audiosr_model: {
          type: "string",
          enum: ["speech", "basic"],
          description: "AudioSR model preset; defaults to speech.",
        },
        audiosr_device: {
          type: "string",
          pattern: "^(?:auto|cpu|mps|cuda(?::\\d{1,3})?)$",
          description: "AudioSR device: auto, cpu, mps, cuda, or cuda:N (up to three digits); defaults to auto.",
        },
        audiosr_ddim_steps: {
          type: "integer",
          minimum: 10,
          maximum: 250,
          description: "AudioSR DDIM steps; defaults to 50.",
        },
        audiosr_guidance_scale: {
          type: "number",
          minimum: 1,
          maximum: 10,
          description: "AudioSR classifier-free guidance scale; defaults to 3.5.",
        },
        audiosr_seed: {
          type: "integer",
          minimum: -2_147_483_648,
          maximum: 2_147_483_647,
          description: "AudioSR random seed; defaults to 42.",
        },
      },
      required: ["request_id", "text", "target"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, openWorldHint: true, destructiveHint: false },
  },
  {
    name: "voiceforge_get_job",
    title: "Get a VoiceForge speech job",
    description: "Poll a VoiceForge synthesis job until it completes, fails, or is cancelled.",
    inputSchema: {
      type: "object",
      properties: { job_id: { type: "string" } },
      required: ["job_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
  },
  {
    name: "voiceforge_get_audio_path",
    title: "Get completed VoiceForge audio path",
    description: "Return the validated, server-controlled local WAV or MP3 path for a completed job so the calling project can copy it into its own assets.",
    inputSchema: {
      type: "object",
      properties: { job_id: { type: "string", pattern: "^vf-(index|vibe|qwen|moss)-[A-Za-z0-9_-]+$" } },
      required: ["job_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, openWorldHint: false, destructiveHint: false },
  },
  {
    name: "voiceforge_cancel_job",
    title: "Cancel a VoiceForge speech job",
    description: "Cancel a queued or running VoiceForge job. Completed and failed jobs are unchanged.",
    inputSchema: {
      type: "object",
      properties: { job_id: { type: "string" } },
      required: ["job_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, openWorldHint: false, destructiveHint: true },
  },
] as const;

export const VOICEFORGE_RESOURCE_TEMPLATES = [
  {
    uriTemplate: "voiceforge://speech/jobs/{jobId}/audio",
    name: "generated-speech-audio",
    description: "Completed VoiceForge speech audio",
  },
  {
    uriTemplate: "voiceforge://default-voices/{voiceId}/audio",
    name: "default-voice-audio",
    description: "VoiceForge default reference voice preview",
  },
] as const;

function log(message: string): void {
  process.stderr.write(`[voiceforge-bridge] ${message}\n`);
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function withTimeout<T>(operation: Promise<T>, milliseconds: number, message: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function parsePort(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  if (!/^\d{1,5}$/u.test(value)) throw new Error("VoiceForge port configuration must be numeric.");
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) throw new Error("VoiceForge port is out of range.");
  return port;
}

const portMin = parsePort(process.env.VOICEFORGE_PORT_MIN, 5000);
const portMax = parsePort(process.env.VOICEFORGE_PORT_MAX, 5020);
if (portMax < portMin || portMax - portMin > 100) throw new Error("VoiceForge port range is invalid.");

function stateDirectory(): string {
  const configured = process.env.VOICEFORGE_STATE_DIR?.trim();
  if (configured) {
    if (!path.isAbsolute(configured)) throw new Error("VOICEFORGE_STATE_DIR must be absolute.");
    return path.resolve(configured);
  }
  const base = process.env.LOCALAPPDATA?.trim()
    || process.env.XDG_STATE_HOME?.trim()
    || path.join(os.homedir(), ".local", "state");
  return path.join(base, "VoiceForge");
}

function resolveVoiceForgeRoot(): string {
  const configured = process.env.VOICEFORGE_ROOT?.trim() || "P:\\_code\\VoiceForge";
  if (!path.isAbsolute(configured) || /["&|<>^%!]/u.test(configured)) {
    throw new Error("VOICEFORGE_ROOT must be a safe absolute path.");
  }
  const resolved = fs.realpathSync.native(configured);
  for (const required of ["VoiceForge.cmd", "package.json"]) {
    if (!fs.statSync(path.join(resolved, required)).isFile()) {
      throw new Error(`VoiceForge installation is missing ${required}.`);
    }
  }
  return resolved;
}

const voiceForgeRoot = resolveVoiceForgeRoot();
const stateRoot = stateDirectory();
const tokenPath = path.join(stateRoot, "mcp-token");
const startupLockPath = path.join(stateRoot, "startup.lock");
const launcherLogPath = path.join(stateRoot, "launcher.log");
const launcherPidPath = path.join(stateRoot, "launcher.pid");
let cachedPort: number | undefined;
let cachedToken: string | undefined;
const legacyNoAuthPorts = new Set<number>();

async function ensureToken(): Promise<string> {
  if (cachedToken) return cachedToken;
  await fsPromises.mkdir(stateRoot, { recursive: true, mode: 0o700 });
  try {
    const token = (await fsPromises.readFile(tokenPath, "utf8")).trim();
    if (!/^[A-Za-z0-9_-]{43}$/u.test(token)) throw new Error("VoiceForge MCP token file is invalid.");
    cachedToken = token;
    return token;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }

  const token = randomBytes(32).toString("base64url");
  try {
    const handle = await fsPromises.open(tokenPath, "wx", 0o600);
    try {
      await handle.writeFile(`${token}\n`, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    return ensureTokenFromExistingFile();
  }
  await fsPromises.chmod(tokenPath, 0o600).catch(() => undefined);
  cachedToken = token;
  return token;
}

async function ensureTokenFromExistingFile(): Promise<string> {
  const deadline = Date.now() + 2_000;
  while (Date.now() < deadline) {
    try {
      const token = (await fsPromises.readFile(tokenPath, "utf8")).trim();
      if (/^[A-Za-z0-9_-]{43}$/u.test(token)) {
        cachedToken = token;
        return token;
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    await sleep(25);
  }
  throw new Error("VoiceForge MCP token file was not initialized safely.");
}

function loopbackFetch(port: number): typeof fetch {
  return async (input, init) => {
    const rawUrl = input instanceof Request ? input.url : input.toString();
    const url = new URL(rawUrl);
    if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || Number(url.port) !== port || url.pathname !== "/mcp") {
      throw new Error("VoiceForge bridge refused a non-loopback MCP request.");
    }
    return fetch(url, { ...init, redirect: "error" });
  };
}

async function connectVerified(port: number, timeoutMs: number, useAuthorization: boolean): Promise<Client> {
  const token = useAuthorization ? await ensureToken() : undefined;
  const client = new Client({ name: "voiceforge-global-bridge", version: BRIDGE_VERSION });
  const transport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${port}/mcp`), {
    requestInit: {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      redirect: "error",
    },
    fetch: loopbackFetch(port),
  });
  try {
    await withTimeout(client.connect(transport), timeoutMs, `Timed out connecting to VoiceForge on port ${port}.`);
    const version = client.getServerVersion();
    const major = Number.parseInt(version?.version?.split(".", 1)[0] || "", 10);
    if (version?.name !== "voiceforge" || major !== REMOTE_MAJOR_VERSION) {
      throw new Error("Endpoint is not a compatible VoiceForge MCP server.");
    }
    const listed = await withTimeout(client.listTools(), timeoutMs, "Timed out validating VoiceForge tools.");
    const names = new Set(listed.tools.map((tool) => tool.name));
    if ([...REQUIRED_REMOTE_TOOLS].some((name) => !names.has(name))) {
      throw new Error("VoiceForge MCP server is missing required tools.");
    }
    return client;
  } catch (error) {
    await client.close().catch(() => undefined);
    throw error;
  }
}

type HealthPreflight = "authenticated" | "no-auth" | "legacy" | "reject";

function secureEqual(left: string, right: string): boolean {
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  return a.length === b.length && timingSafeEqual(a, b);
}

async function preflightPort(port: number): Promise<HealthPreflight> {
  const token = await ensureToken();
  const nonce = randomBytes(24).toString("base64url");
  const url = new URL(`http://127.0.0.1:${port}/api/voiceforge/health`);
  url.searchParams.set("nonce", nonce);
  try {
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      redirect: "error",
      signal: AbortSignal.timeout(DISCOVERY_TIMEOUT_MS),
    });
    if (!response.ok || !response.headers.get("content-type")?.toLowerCase().includes("application/json")) {
      return "legacy";
    }
    const raw = await response.text();
    if (raw.length > 16_384) return "reject";
    const health = JSON.parse(raw) as Record<string, unknown>;
    const major = Number.parseInt(String(health.version || "").split(".", 1)[0], 10);
    if (
      health.service !== "voiceforge" ||
      major !== REMOTE_MAJOR_VERSION ||
      health.mcpPath !== "/mcp" ||
      health.port !== port
    ) {
      return "reject";
    }
    if (health.authRequired !== true) return "no-auth";
    if (typeof health.authProof !== "string") return "reject";
    const expected = createHmac("sha256", token)
      .update("voiceforge-health-v1\0", "utf8")
      .update(nonce, "utf8")
      .digest("base64url");
    return secureEqual(health.authProof, expected) ? "authenticated" : "reject";
  } catch {
    return "legacy";
  }
}

async function inspectPort(port: number): Promise<boolean> {
  let client: Client | undefined;
  try {
    const preflight = await preflightPort(port);
    if (preflight === "reject") return false;
    const useAuthorization = preflight === "authenticated";
    client = await connectVerified(port, DISCOVERY_TIMEOUT_MS, useAuthorization);
    if (useAuthorization) legacyNoAuthPorts.delete(port);
    else legacyNoAuthPorts.add(port);
    return true;
  } catch {
    return false;
  } finally {
    if (client) await client.close().catch(() => undefined);
  }
}

async function discoverVoiceForge(): Promise<number | undefined> {
  const ports = Array.from({ length: portMax - portMin + 1 }, (_value, index) => portMin + index);
  const inspected = await Promise.all(ports.map(async (port) => ({ port, ready: await inspectPort(port) })));
  return inspected.find((item) => item.ready)?.port;
}

async function portIsFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.once("error", () => resolve(false));
    server.listen({ host: "127.0.0.1", port, exclusive: true }, () => {
      server.close(() => resolve(true));
    });
  });
}

function processIsAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

async function selectLaunchPort(): Promise<number> {
  for (let port = portMin; port <= portMax; port += 1) {
    if (await portIsFree(port)) return port;
  }
  throw new Error(`No free local port is available from ${portMin} through ${portMax}.`);
}

function sanitizedLaunchEnvironment(token: string, port: number): NodeJS.ProcessEnv {
  const allowExact = new Set([
    "ALLUSERSPROFILE", "ALL_PROXY", "APPDATA", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "COMSPEC", "CURL_CA_BUNDLE", "HF_HOME", "HF_HUB_CACHE",
    "HOME", "HTTP_PROXY", "HTTPS_PROXY", "LANG", "LC_ALL", "LOCALAPPDATA",
    "NODE_EXTRA_CA_CERTS", "NO_PROXY", "NUMBER_OF_PROCESSORS", "OS", "PATHEXT",
    "PROCESSOR_ARCHITECTURE", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_DIR", "SSL_CERT_FILE", "SYSTEMDRIVE", "SYSTEMROOT",
    "TEMP", "TMP", "TMPDIR", "TRANSFORMERS_CACHE", "USERPROFILE",
    "VOICEFORGE_AUDIOSR_BIN", "VOICEFORGE_FFMPEG_BIN", "WINDIR", "XDG_CACHE_HOME",
  ]);
  const allowPrefixes = ["CUDA_", "HSA_", "KMP_", "MOSS_TTS_", "NVIDIA_", "OMP_", "PYTORCH_", "QWEN_TTS_", "ROCR_", "TORCH_"];
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    const upper = key.toUpperCase();
    if (allowExact.has(upper) || allowPrefixes.some((prefix) => upper.startsWith(prefix))) {
      // Windows environment keys are case-insensitive. Canonicalize them so
      // Start-Process never receives both Path and PATH.
      env[upper] = value;
    }
  }
  const safePath = (process.env.PATH || "")
    .split(path.delimiter)
    .filter((entry) => entry && path.isAbsolute(entry) && fs.existsSync(entry));
  env.PATH = [...new Set(safePath)].join(path.delimiter);
  env.HOST = "127.0.0.1";
  env.PORT = String(port);
  env.VOICEFORGE_NO_PAUSE = "1";
  env.VOICEFORGE_MCP_BEARER_TOKEN = token;
  env.VOICEFORGE_STATE_DIR = stateRoot;
  return env;
}

async function rotateLauncherLog(): Promise<void> {
  for (const logPath of [launcherLogPath, `${launcherLogPath}.stdout.log`, `${launcherLogPath}.stderr.log`]) {
    try {
      const stat = await fsPromises.stat(logPath);
      if (stat.size > 5 * 1024 * 1024) {
        await fsPromises.rm(`${logPath}.1`, { force: true });
        await fsPromises.rename(logPath, `${logPath}.1`);
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
}

async function launchVoiceForge(port: number): Promise<number> {
  if (process.platform !== "win32") throw new Error("This VoiceForge installation is configured for Windows.");
  const token = await ensureToken();
  await rotateLauncherLog();
  const systemRoot = process.env.SystemRoot || process.env.SYSTEMROOT || "C:\\Windows";
  const command = path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const wrapper = path.join(path.dirname(fileURLToPath(import.meta.url)), "start-voiceforge.ps1");
  if (!fs.statSync(wrapper).isFile()) throw new Error("VoiceForge launcher wrapper is missing from the plugin.");
  await fsPromises.rm(launcherPidPath, { force: true });
  const logFd = fs.openSync(launcherLogPath, "a", 0o600);
  try {
    const child = spawn(command, [
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      wrapper,
      "-VoiceForgeRoot",
      voiceForgeRoot,
      "-Port",
      String(port),
      "-LauncherLogPath",
      launcherLogPath,
      "-ProcessIdPath",
      launcherPidPath,
    ], {
      cwd: voiceForgeRoot,
      detached: false,
      env: sanitizedLaunchEnvironment(token, port),
      windowsHide: true,
      stdio: ["ignore", logFd, logFd],
    });
    await new Promise<void>((resolve, reject) => {
      child.once("error", reject);
      child.once("close", (code) => {
        if (code === 0) resolve();
        else reject(new Error(`VoiceForge launcher wrapper exited with code ${code ?? "unknown"}.`));
      });
    });
  } finally {
    fs.closeSync(logFd);
  }
  const launchedPid = Number((await fsPromises.readFile(launcherPidPath, "utf8")).trim());
  if (!Number.isInteger(launchedPid) || launchedPid <= 0) {
    throw new Error("VoiceForge launcher did not report a valid process ID.");
  }
  return launchedPid;
}

type StartupLock = { handle: fsPromises.FileHandle; ownerId: string };

async function tryAcquireStartupLock(): Promise<StartupLock | undefined> {
  await fsPromises.mkdir(stateRoot, { recursive: true, mode: 0o700 });
  try {
    const handle = await fsPromises.open(startupLockPath, "wx", 0o600);
    const ownerId = randomBytes(16).toString("base64url");
    await handle.writeFile(JSON.stringify({ pid: process.pid, ownerId, createdAt: Date.now() }), "utf8");
    await handle.sync();
    return { handle, ownerId };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    try {
      const stat = await fsPromises.stat(startupLockPath);
      const age = Date.now() - stat.mtimeMs;
      let ownerPid = 0;
      try {
        const lock = JSON.parse(await fsPromises.readFile(startupLockPath, "utf8")) as { pid?: unknown };
        ownerPid = typeof lock.pid === "number" ? lock.pid : 0;
      } catch {}
      if ((ownerPid > 0 && !processIsAlive(ownerPid)) || (ownerPid === 0 && age > 5_000) || age > STARTUP_LOCK_STALE_MS) {
        await fsPromises.rm(startupLockPath, { force: true });
      }
    } catch {}
    return undefined;
  }
}

async function releaseStartupLock(lock: StartupLock): Promise<void> {
  await lock.handle.close().catch(() => undefined);
  try {
    const current = JSON.parse(await fsPromises.readFile(startupLockPath, "utf8")) as { ownerId?: unknown };
    if (current.ownerId === lock.ownerId) await fsPromises.rm(startupLockPath, { force: true });
  } catch {}
}

async function ensureVoiceForgePort(): Promise<number> {
  if (cachedPort && await inspectPort(cachedPort)) return cachedPort;
  cachedPort = undefined;
  const existing = await discoverVoiceForge();
  if (existing) {
    cachedPort = existing;
    return existing;
  }

  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const lock = await tryAcquireStartupLock();
    if (lock) {
      try {
        const racedInstance = await discoverVoiceForge();
        if (racedInstance) {
          cachedPort = racedInstance;
          return racedInstance;
        }
        const port = await selectLaunchPort();
        log(`Starting VoiceForge on 127.0.0.1:${port}; launcher output is under ${stateRoot}.`);
        const launchedPid = await launchVoiceForge(port);
        while (Date.now() < deadline) {
          if (await inspectPort(port)) {
            cachedPort = port;
            return port;
          }
          if (!processIsAlive(launchedPid)) {
            throw new Error(`VoiceForge exited before becoming ready. Check ${stateRoot}.`);
          }
          await sleep(1_000);
        }
        throw new Error(`VoiceForge did not become ready. Check ${stateRoot}.`);
      } finally {
        await releaseStartupLock(lock);
      }
    }

    const running = await discoverVoiceForge();
    if (running) {
      cachedPort = running;
      return running;
    }
    await sleep(1_000);
  }
  throw new Error(`Timed out waiting for VoiceForge. Check ${stateRoot}.`);
}

async function openRemoteClient(): Promise<Client> {
  const port = await ensureVoiceForgePort();
  try {
    return await connectVerified(port, 15_000, !legacyNoAuthPorts.has(port));
  } catch {
    cachedPort = undefined;
    const recoveredPort = await ensureVoiceForgePort();
    return connectVerified(recoveredPort, 15_000, !legacyNoAuthPorts.has(recoveredPort));
  }
}

async function remoteToolCall(name: string, args: Record<string, unknown>): Promise<any> {
  const client = await openRemoteClient();
  try {
    return await client.callTool(
      { name, arguments: args },
      undefined,
      { timeout: REMOTE_OPERATION_TIMEOUT_MS }
    );
  } catch (error) {
    if (MUTATING_REMOTE_TOOLS.has(name)) {
      throw new Error(
        `VoiceForge lost the response to ${name}; the bridge will not replay a potentially accepted synthesis/cancellation request. ` +
        `Check the job with the same request ID before retrying. Original error: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    throw error;
  } finally {
    await client.close().catch(() => undefined);
  }
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

export async function completedAudioPath(jobId: string): Promise<{
  path: string;
  size: number;
  format: "wav" | "mp3";
  mimeType: "audio/wav" | "audio/mpeg";
}> {
  const match = /^vf-(index|vibe|qwen|moss)-([A-Za-z0-9_-]+)$/u.exec(jobId);
  if (!match) throw new Error("VoiceForge job ID is invalid.");
  const status = await remoteToolCall("voiceforge_get_job", { job_id: jobId });
  const publicJob = status?.structuredContent?.job;
  if (!publicJob || publicJob.job_id !== jobId || publicJob.status !== "completed") {
    throw new Error("VoiceForge audio is not completed yet.");
  }
  const format =
    publicJob.output_format === undefined || publicJob.output_format === "wav"
      ? "wav"
      : publicJob.output_format === "mp3"
        ? "mp3"
        : undefined;
  if (!format) throw new Error("VoiceForge returned an unsupported completed audio format.");
  const mimeType = format === "mp3" ? "audio/mpeg" : "audio/wav";
  const roots: Record<string, string> = {
    index: path.join(voiceForgeRoot, "attached_assets", "index-tts", "jobs"),
    vibe: path.join(voiceForgeRoot, "attached_assets", "vibevoice", "jobs"),
    qwen: path.join(voiceForgeRoot, "attached_assets", "qwen3-tts", "jobs"),
    moss: path.join(voiceForgeRoot, "attached_assets", "moss-tts-v1.5", "jobs"),
  };
  const jobsRoot = fs.realpathSync.native(roots[match[1]]);
  const output = fs.realpathSync.native(path.join(jobsRoot, match[2], `output.${format}`));
  if (!isWithin(jobsRoot, output)) throw new Error("VoiceForge audio resolved outside its managed jobs directory.");
  const stat = await fsPromises.stat(output);
  if (!stat.isFile() || stat.size <= 0 || stat.size > MAX_LOCAL_AUDIO_BYTES) {
    throw new Error("VoiceForge audio file is missing or exceeds the safety limit.");
  }
  return { path: output, size: stat.size, format, mimeType };
}

function errorToolResult(error: unknown) {
  return {
    isError: true,
    content: [{ type: "text" as const, text: error instanceof Error ? error.message : "VoiceForge operation failed." }],
  };
}

export function createBridgeServer(): Server {
  const server = new Server(
    { name: "voiceforge-global", version: BRIDGE_VERSION },
    {
      capabilities: { tools: {}, resources: {} },
      instructions:
        "VoiceForge turns text into speech and starts automatically. Draft narration with the current chat model when requested, then recommend a VoiceForge model. Require an explicit Local or Agent target before synthesis. Poll jobs to completion; use voiceforge_get_audio_path for local video/project asset handoff.",
    }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [...VOICEFORGE_TOOLS] }));
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = (request.params.arguments || {}) as Record<string, unknown>;
    try {
      if (name === "voiceforge_get_audio_path") {
        const jobId = typeof args.job_id === "string" ? args.job_id : "";
        const audio = await completedAudioPath(jobId);
        return {
          structuredContent: {
            job_id: jobId,
            audio_path: audio.path,
            size_bytes: audio.size,
            output_format: audio.format,
            mime_type: audio.mimeType,
          },
          content: [{
            type: "text" as const,
            text: `Completed VoiceForge ${audio.format.toUpperCase()}: ${audio.path}`,
          }],
        };
      }
      if (!REQUIRED_REMOTE_TOOLS.has(name)) return errorToolResult(new Error(`Unknown VoiceForge tool: ${name}`));
      return await remoteToolCall(name, args);
    } catch (error) {
      return errorToolResult(error);
    }
  });
  server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: [] }));
  server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({
    resourceTemplates: [...VOICEFORGE_RESOURCE_TEMPLATES],
  }));
  server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
    const client = await openRemoteClient();
    try {
      return await client.readResource(request.params, { timeout: REMOTE_OPERATION_TIMEOUT_MS });
    } finally {
      await client.close().catch(() => undefined);
    }
  });
  return server;
}

export async function main(): Promise<void> {
  const server = createBridgeServer();
  const transport = new StdioServerTransport();
  const shutdown = async () => {
    await server.close().catch(() => undefined);
  };
  process.once("SIGINT", () => void shutdown().finally(() => process.exit(0)));
  process.once("SIGTERM", () => void shutdown().finally(() => process.exit(0)));
  await server.connect(transport);
}

const entryPoint = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : "";
if (entryPoint === import.meta.url) {
  main().catch((error) => {
    log(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
