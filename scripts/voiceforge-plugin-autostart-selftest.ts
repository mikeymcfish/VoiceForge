import assert from "node:assert/strict";
import fs from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const bridge = path.join(projectRoot, "codex-plugin", "voiceforge", "scripts", "voiceforge-mcp-bridge.mjs");
const sandbox = await fs.mkdtemp(path.join(os.tmpdir(), "voiceforge-autostart-"));
const fakeRoot = path.join(sandbox, "fake-root");
const stateRoot = path.join(sandbox, "state");
const callerA = path.join(sandbox, "caller-a");
const callerB = path.join(sandbox, "caller-b");
await Promise.all([fakeRoot, stateRoot, callerA, callerB].map((directory) => fs.mkdir(directory, { recursive: true })));
await fs.writeFile(
  path.join(stateRoot, "startup.lock"),
  JSON.stringify({ pid: 2_147_483_647, ownerId: "orphaned-test-lock", createdAt: Date.now() }),
  "utf8"
);

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen({ host: "127.0.0.1", port: 0 }, () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

const port = await freePort();
const requiredTools = [
  "voiceforge_list_models",
  "voiceforge_list_voices",
  "voiceforge_recommend_model",
  "voiceforge_generate_speech",
  "voiceforge_get_job",
  "voiceforge_cancel_job",
];
const fakeServer = `
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const args = process.argv.slice(2);
const portIndex = args.indexOf("--port");
const port = Number(args[portIndex + 1]);
const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\\/(?:[A-Za-z]:)/u, (value) => value.slice(1)));
const tools = ${JSON.stringify(requiredTools)}.map((name) => ({ name, description: name, inputSchema: { type: "object", properties: {} } }));
function send(res, payload) {
  const body = "event: message\\ndata: " + JSON.stringify(payload) + "\\n\\n";
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
  res.end(body);
}
const server = http.createServer((req, res) => {
  if (req.method !== "POST" || req.url !== "/mcp") { res.writeHead(405).end(); return; }
  let body = "";
  req.setEncoding("utf8");
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    const message = JSON.parse(body);
    if (message.id === undefined) { res.writeHead(202).end(); return; }
    if (message.method === "initialize") {
      send(res, { jsonrpc: "2.0", id: message.id, result: { protocolVersion: message.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: "voiceforge", version: "2.0.0" } } });
      return;
    }
    if (message.method === "tools/list") { send(res, { jsonrpc: "2.0", id: message.id, result: { tools } }); return; }
    if (message.method === "tools/call" && message.params.name === "voiceforge_list_models") {
      const models = Array.from({ length: 6 }, (_value, index) => ({ id: "model-" + index }));
      send(res, { jsonrpc: "2.0", id: message.id, result: { content: [{ type: "text", text: "six models" }], structuredContent: { models } } });
      return;
    }
    send(res, { jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "unsupported" } });
  });
});
server.listen(port, "127.0.0.1", () => fs.writeFileSync(path.join(root, "pid.txt"), String(process.pid)));
`;
await fs.writeFile(path.join(fakeRoot, "fake-server.mjs"), fakeServer, "utf8");
await fs.writeFile(path.join(fakeRoot, "package.json"), '{"name":"voiceforge-autostart-fixture","version":"1.0.0"}\n', "utf8");
await fs.writeFile(
  path.join(fakeRoot, "VoiceForge.cmd"),
  '@echo off\r\necho launch>>"%~dp0launches.txt"\r\nnode "%~dp0fake-server.mjs" %*\r\n',
  "utf8"
);

function bridgeClient(cwd: string): { client: Client; transport: StdioClientTransport } {
  const client = new Client({ name: "voiceforge-autostart-selftest", version: "1.0.0" });
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [bridge],
    cwd,
    env: {
      ...Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined)),
      VOICEFORGE_ROOT: fakeRoot,
      VOICEFORGE_PORT_MIN: String(port),
      VOICEFORGE_PORT_MAX: String(port),
      VOICEFORGE_STATE_DIR: stateRoot,
    },
    stderr: "pipe",
  });
  return { client, transport };
}

const a = bridgeClient(callerA);
const b = bridgeClient(callerB);
let bridgeDiagnostics = "";
a.transport.stderr?.on("data", (chunk) => { bridgeDiagnostics += `[a] ${chunk.toString()}`; });
b.transport.stderr?.on("data", (chunk) => { bridgeDiagnostics += `[b] ${chunk.toString()}`; });
let fakePid: number | undefined;
let failed = false;
try {
  await Promise.all([a.client.connect(a.transport), b.client.connect(b.transport)]);
  const [resultA, resultB] = await Promise.all([
    a.client.callTool({ name: "voiceforge_list_models", arguments: {} }, undefined, { timeout: 60_000 }),
    b.client.callTool({ name: "voiceforge_list_models", arguments: {} }, undefined, { timeout: 60_000 }),
  ]);
  assert.equal((resultA.structuredContent as any)?.models?.length, 6);
  assert.equal((resultB.structuredContent as any)?.models?.length, 6);
  const launches = (await fs.readFile(path.join(fakeRoot, "launches.txt"), "utf8")).trim().split(/\r?\n/u);
  assert.equal(launches.length, 1, "Concurrent bridges launched VoiceForge more than once.");
  fakePid = Number(await fs.readFile(path.join(fakeRoot, "pid.txt"), "utf8"));
  assert.deepEqual(await fs.readdir(callerA), []);
  assert.deepEqual(await fs.readdir(callerB), []);
} catch (error) {
  failed = true;
  const launcherLog = await fs.readFile(path.join(stateRoot, "launcher.log"), "utf8").catch(() => "<no launcher log>");
  const fakeEntries = await fs.readdir(fakeRoot).catch(() => []);
  console.error(`Bridge diagnostics:\n${bridgeDiagnostics}\nLauncher log:\n${launcherLog}\nFake root entries: ${fakeEntries.join(", ")}`);
  throw error;
} finally {
  await Promise.all([a.client.close().catch(() => undefined), b.client.close().catch(() => undefined)]);
  if (fakePid && Number.isInteger(fakePid)) {
    try { process.kill(fakePid); } catch {}
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  if (failed) console.error(`Preserved failed fixture: ${sandbox}`);
  else await fs.rm(sandbox, { recursive: true, force: true });
}

console.log("VoiceForge plugin auto-start and concurrent launch-lock self-test passed.");
