import type { Express, Request, Response } from "express";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createVoiceForgeMcpServer } from "./mcp-server";
import { voiceForgeOrchestrator } from "./voiceforge-orchestrator";
import {
  getVoiceForgeMcpBearerToken,
  voiceForgeMcpAuthRequired,
  voiceForgeMcpBearerAuthorized,
  voiceForgeMcpHealthProof,
} from "./mcp-auth";

function jsonRpcError(res: Response, status: number, message: string): void {
  res.status(status).json({
    jsonrpc: "2.0",
    error: { code: status === 401 ? -32001 : -32000, message },
    id: null,
  });
}

function bearerAuthorized(req: Request): boolean {
  return voiceForgeMcpBearerAuthorized(req.get("authorization"));
}

function remoteMcpEnabled(): boolean {
  const host = (process.env.HOST || "127.0.0.1").trim().toLowerCase();
  const loopback = host === "127.0.0.1" || host === "localhost" || host === "::1";
  if (loopback) return true;
  const publicUrl = process.env.VOICEFORGE_PUBLIC_URL?.trim();
  return (
    /^(1|true|yes|on)$/i.test(process.env.VOICEFORGE_ENABLE_REMOTE_MCP || "") &&
    Boolean(getVoiceForgeMcpBearerToken()) &&
    Boolean(publicUrl && /^https:\/\//iu.test(publicUrl))
  );
}

function publicBaseUrl(req: Request): string {
  const configured = process.env.VOICEFORGE_PUBLIC_URL?.trim();
  if (configured) return configured.replace(/\/+$/u, "");
  return `http://127.0.0.1:${process.env.PORT || "5000"}`;
}

export function voiceForgeMcpHealth(nonce?: string) {
  return {
    service: "voiceforge",
    version: "2.0.0",
    pid: process.pid,
    port: Number.parseInt(process.env.PORT || "5000", 10),
    mcpPath: "/mcp",
    authRequired: voiceForgeMcpAuthRequired(),
    authProof: voiceForgeMcpHealthProof(nonce),
  };
}

export function registerVoiceForgeMcpRoutes(app: Express): void {
  app.post("/mcp", async (req, res) => {
    if (!remoteMcpEnabled()) {
      return jsonRpcError(
        res,
        403,
        "Remote MCP is disabled. Keep VoiceForge on loopback or set VOICEFORGE_ENABLE_REMOTE_MCP=true deliberately."
      );
    }
    if (!bearerAuthorized(req)) return jsonRpcError(res, 401, "Invalid VoiceForge MCP bearer token.");

    const server = createVoiceForgeMcpServer({ publicBaseUrl: publicBaseUrl(req) });
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    try {
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (error) {
      console.error("VoiceForge MCP request failed:", error);
      if (!res.headersSent) jsonRpcError(res, 500, "VoiceForge MCP request failed.");
    } finally {
      await transport.close().catch(() => undefined);
      await server.close().catch(() => undefined);
    }
  });

  for (const method of ["get", "delete"] as const) {
    app[method]("/mcp", (_req, res) => {
      jsonRpcError(res, 405, "VoiceForge uses stateless Streamable HTTP; send MCP JSON-RPC requests with POST.");
    });
  }

  app.get("/api/mcp/speech/jobs/:id/audio", (req, res) => {
    if (!remoteMcpEnabled()) {
      return res.status(403).json({ error: "VoiceForge MCP is not enabled for this bind configuration." });
    }
    if (!bearerAuthorized(req)) return res.status(401).json({ error: "Invalid VoiceForge MCP bearer token." });
    const job = voiceForgeOrchestrator.getJob(req.params.id);
    const outputPath = voiceForgeOrchestrator.getJobOutputPath(req.params.id);
    if (!job || job.status !== "completed" || !outputPath) {
      return res.status(404).json({ error: "Generated audio is not ready" });
    }
    res.type("audio/wav");
    const disposition = req.query.download === "1" ? "attachment" : "inline";
    res.setHeader("Content-Disposition", `${disposition}; filename="${job.model}-${job.id}.wav"`);
    res.sendFile(outputPath);
  });
}
