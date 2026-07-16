import { createHmac, timingSafeEqual } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

function tokenFilePath(): string {
  const configured = process.env.VOICEFORGE_STATE_DIR?.trim();
  if (configured && path.isAbsolute(configured)) return path.join(configured, "mcp-token");
  const base = process.env.LOCALAPPDATA?.trim()
    || process.env.XDG_STATE_HOME?.trim()
    || path.join(os.homedir(), ".local", "state");
  return path.join(base, "VoiceForge", "mcp-token");
}

type TokenState = { provisioned: boolean; token?: string };

function loadTokenState(): TokenState {
  const fromEnvironment = process.env.VOICEFORGE_MCP_BEARER_TOKEN?.trim();
  if (fromEnvironment) return { provisioned: true, token: fromEnvironment };
  try {
    const token = fs.readFileSync(tokenFilePath(), "utf8").trim();
    return /^[A-Za-z0-9_-]{32,128}$/u.test(token)
      ? { provisioned: true, token }
      : { provisioned: true };
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "ENOENT"
      ? { provisioned: false }
      : { provisioned: true };
  }
}

export function getVoiceForgeMcpBearerToken(): string | undefined {
  return loadTokenState().token;
}

export function voiceForgeMcpAuthRequired(): boolean {
  return loadTokenState().provisioned;
}

export function voiceForgeMcpBearerAuthorized(header: string | undefined): boolean {
  const state = loadTokenState();
  if (!state.provisioned) return true;
  if (!state.token) return false;
  const expected = state.token;
  const actual = header?.startsWith("Bearer ") ? header.slice(7) : "";
  const expectedBuffer = Buffer.from(expected);
  const actualBuffer = Buffer.from(actual);
  return actualBuffer.length === expectedBuffer.length && timingSafeEqual(actualBuffer, expectedBuffer);
}

export function voiceForgeMcpHealthProof(nonce: string | undefined): string | undefined {
  if (!nonce || !/^[A-Za-z0-9_-]{16,128}$/u.test(nonce)) return undefined;
  const token = loadTokenState().token;
  if (!token) return undefined;
  return createHmac("sha256", token)
    .update("voiceforge-health-v1\0", "utf8")
    .update(nonce, "utf8")
    .digest("base64url");
}
