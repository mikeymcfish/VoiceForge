import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let bridge = path.join(root, "codex-plugin", "voiceforge", "scripts", "voiceforge-mcp-bridge.mjs");
if (process.argv.includes("--installed")) {
  const cacheRoot = path.join(os.homedir(), ".codex", "plugins", "cache", "personal", "voiceforge");
  const versions = await Promise.all((await fs.readdir(cacheRoot)).map(async (name) => ({
    name,
    modified: (await fs.stat(path.join(cacheRoot, name))).mtimeMs,
  })));
  versions.sort((a, b) => b.modified - a.modified);
  assert.ok(versions[0], "No installed VoiceForge plugin cache was found.");
  bridge = path.join(cacheRoot, versions[0].name, "scripts", "voiceforge-mcp-bridge.mjs");
}
const unrelated = await fs.mkdtemp(path.join(os.tmpdir(), "voiceforge-global-plugin-"));
const stateDirectory = await fs.mkdtemp(path.join(os.tmpdir(), "voiceforge-global-plugin-state-"));
const initialEntries = await fs.readdir(unrelated);
const client = new Client({ name: "voiceforge-global-plugin-selftest", version: "1.0.0" });
const transport = new StdioClientTransport({
  command: process.execPath,
  args: [bridge],
  cwd: unrelated,
  env: {
    ...Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined)),
    VOICEFORGE_ROOT: root,
    VOICEFORGE_PORT_MIN: "5000",
    VOICEFORGE_PORT_MAX: "5020",
    VOICEFORGE_STATE_DIR: stateDirectory,
  },
  stderr: "pipe",
});
let diagnostics = "";
transport.stderr?.on("data", (chunk) => { diagnostics += chunk.toString(); });

try {
  const startedAt = Date.now();
  await client.connect(transport);
  const tools = await client.listTools();
  assert.ok(Date.now() - startedAt < 5_000, "The bridge must initialize and list static tools without waiting for VoiceForge startup.");
  assert.deepEqual(
    tools.tools.map((tool) => tool.name).sort(),
    [
      "voiceforge_cancel_job",
      "voiceforge_generate_speech",
      "voiceforge_get_audio_path",
      "voiceforge_get_job",
      "voiceforge_list_models",
      "voiceforge_list_voices",
      "voiceforge_recommend_model",
    ]
  );

  const listed = await client.callTool(
    { name: "voiceforge_list_models", arguments: {} },
    undefined,
    { timeout: 6 * 60_000 }
  );
  assert.equal(listed.isError, undefined, JSON.stringify(listed.content));
  assert.equal((listed.structuredContent as any)?.models?.length, 6);
  assert.deepEqual(await fs.readdir(unrelated), initialEntries, "The bridge wrote into the unrelated calling project.");
  assert.doesNotMatch(diagnostics, /Bearer\s+[A-Za-z0-9_-]+/u, "The bridge leaked a bearer token.");
} finally {
  await client.close().catch(() => undefined);
  await fs.rm(unrelated, { recursive: true, force: true });
  await fs.rm(stateDirectory, { recursive: true, force: true });
}

console.log("VoiceForge global plugin bridge self-test passed.");
