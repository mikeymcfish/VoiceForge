import assert from "node:assert/strict";
import { createHmac, randomBytes } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  getVoiceForgeMcpBearerToken,
  voiceForgeMcpAuthRequired,
  voiceForgeMcpBearerAuthorized,
  voiceForgeMcpHealthProof,
} from "../server/mcp-auth";

const originalState = process.env.VOICEFORGE_STATE_DIR;
const originalToken = process.env.VOICEFORGE_MCP_BEARER_TOKEN;
const state = await fs.mkdtemp(path.join(os.tmpdir(), "voiceforge-mcp-auth-"));
try {
  process.env.VOICEFORGE_STATE_DIR = state;
  delete process.env.VOICEFORGE_MCP_BEARER_TOKEN;
  assert.equal(getVoiceForgeMcpBearerToken(), undefined);
  assert.equal(voiceForgeMcpAuthRequired(), false);
  assert.equal(voiceForgeMcpBearerAuthorized(undefined), true);

  const fileToken = randomBytes(32).toString("base64url");
  await fs.writeFile(path.join(state, "mcp-token"), `${fileToken}\n`, { encoding: "utf8", mode: 0o600 });
  assert.equal(getVoiceForgeMcpBearerToken(), fileToken);
  assert.equal(voiceForgeMcpAuthRequired(), true);
  assert.equal(voiceForgeMcpBearerAuthorized(`Bearer ${fileToken}`), true);
  assert.equal(voiceForgeMcpBearerAuthorized("Bearer wrong"), false);
  assert.equal(voiceForgeMcpBearerAuthorized(undefined), false);
  const nonce = randomBytes(24).toString("base64url");
  const expectedProof = createHmac("sha256", fileToken)
    .update("voiceforge-health-v1\0", "utf8")
    .update(nonce, "utf8")
    .digest("base64url");
  assert.equal(voiceForgeMcpHealthProof(nonce), expectedProof);

  const environmentToken = randomBytes(32).toString("base64url");
  process.env.VOICEFORGE_MCP_BEARER_TOKEN = environmentToken;
  assert.equal(getVoiceForgeMcpBearerToken(), environmentToken);
  assert.equal(voiceForgeMcpBearerAuthorized(`Bearer ${environmentToken}`), true);
  assert.equal(voiceForgeMcpBearerAuthorized(`Bearer ${fileToken}`), false);

  delete process.env.VOICEFORGE_MCP_BEARER_TOKEN;
  await fs.writeFile(path.join(state, "mcp-token"), "corrupt\n", "utf8");
  assert.equal(getVoiceForgeMcpBearerToken(), undefined);
  assert.equal(voiceForgeMcpAuthRequired(), true);
  assert.equal(voiceForgeMcpBearerAuthorized(undefined), false);
  assert.equal(voiceForgeMcpHealthProof(nonce), undefined);
} finally {
  if (originalState === undefined) delete process.env.VOICEFORGE_STATE_DIR;
  else process.env.VOICEFORGE_STATE_DIR = originalState;
  if (originalToken === undefined) delete process.env.VOICEFORGE_MCP_BEARER_TOKEN;
  else process.env.VOICEFORGE_MCP_BEARER_TOKEN = originalToken;
  await fs.rm(state, { recursive: true, force: true });
}

console.log("VoiceForge MCP authentication self-test passed.");
