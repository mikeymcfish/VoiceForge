import 'dotenv/config';
import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { evaluateLocalRequest, isLoopbackBindHost } from "./loopback-request-policy";

const port = parseInt(process.env.PORT || '5000', 10);
const host = process.env.HOST || "127.0.0.1";
const app = express();
// Disable ETag on API responses to avoid 304 for frequently polled endpoints
app.set("etag", false);
app.disable("x-powered-by");
app.use((_req, res, next) => {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  next();
});
if (isLoopbackBindHost(host)) {
  app.use((req, res, next) => {
    const policy = evaluateLocalRequest(host, port, req.get("host"), req.get("origin"));
    if (policy === "invalid-host") return res.status(421).json({ error: "Invalid local VoiceForge host." });
    if (policy === "invalid-origin") return res.status(403).json({ error: "Cross-origin local VoiceForge request blocked." });
    next();
  });
}
// Basic no-cache headers for API routes
app.use((req, res, next) => {
  if (req.path.startsWith("/api/")) {
    res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
  }
  next();
});
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: false, limit: '20mb' }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (res.statusCode >= 400 && capturedJsonResponse) {
        const detail = capturedJsonResponse.error ?? capturedJsonResponse.message;
        if (typeof detail === "string") logLine += ` :: ${detail}`;
      }

      if (logLine.length > 160) {
        logLine = logLine.slice(0, 159) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
    if (res.headersSent) return next(err);
    const isUploadError = err?.name === "MulterError";
    const status = isUploadError
      ? err?.code === "LIMIT_FILE_SIZE" || err?.code === "LIMIT_FILE_COUNT" ? 413 : 400
      : err.status || err.statusCode || 500;
    const publicMessage = status >= 500 && app.get("env") === "production"
      ? "Internal Server Error"
      : err.message || "Internal Server Error";

    if (status >= 500) console.error(err);
    res.status(status).json({ error: publicMessage });
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // VoiceForge can install and execute local model runtimes, so local-only is
  // the safe default. Hosted environments may opt in with HOST=0.0.0.0.
  const listenOptions: { port: number; host: string; reusePort?: boolean } = {
    port,
    host,
  };
  if (process.env.REUSE_PORT === "true") {
    listenOptions.reusePort = true;
  }

  server.listen(listenOptions, () => {
    log(`VoiceForge ready at http://${host}:${port}`);
  });
})();
