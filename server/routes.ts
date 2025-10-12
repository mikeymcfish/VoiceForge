import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { WebSocketServer, WebSocket } from "ws";
import AdmZip from "adm-zip";
import { parse } from "node-html-parser";
import { textProcessor } from "./text-processor";
import { llmService } from "./llm-service";
import { nanoid } from "nanoid";
import type { ProcessingConfig, LogEntry, WSMessage } from "@shared/schema";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
});

async function parseEpub(buffer: Buffer): Promise<string> {
  try {
    const zip = new AdmZip(buffer);
    const zipEntries = zip.getEntries();
    
    const textParts: string[] = [];
    
    // Find and parse all HTML/XHTML files in the EPUB
    for (const entry of zipEntries) {
      const fileName = entry.entryName.toLowerCase();
      
      // Look for content files (HTML/XHTML in OEBPS or similar directories)
      if ((fileName.endsWith('.html') || fileName.endsWith('.xhtml') || fileName.endsWith('.htm')) &&
          !fileName.includes('nav.') && !fileName.includes('toc.')) {
        
        try {
          const content = entry.getData().toString('utf-8');
          const root = parse(content);
          
          // Extract text from body
          const body = root.querySelector('body');
          if (body) {
            const text = body.textContent
              .replace(/\s+/g, ' ') // Normalize whitespace
              .trim();
            
            if (text.length > 0) {
              textParts.push(text);
            }
          }
        } catch (err) {
          console.error(`Error parsing entry ${entry.entryName}:`, err);
        }
      }
    }
    
    if (textParts.length === 0) {
      throw new Error("No readable content found in EPUB file");
    }
    
    return textParts.join('\n\n');
  } catch (error) {
    console.error("EPUB parsing error:", error);
    throw new Error("Failed to parse EPUB file: " + (error instanceof Error ? error.message : "Unknown error"));
  }
}

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);

  // Prompt preview endpoint
  app.post("/api/preview-prompts", async (req, res) => {
    try {
      const { sampleText, config } = req.body as {
        sampleText: string;
        config: ProcessingConfig;
      };

      if (!sampleText || !config) {
        return res.status(400).json({ error: "Missing sample text or configuration" });
      }

      const prompts = llmService.getPromptPreviews(
        sampleText,
        config.cleaningOptions,
        config.speakerConfig,
        config.customInstructions
      );

      res.json(prompts);
    } catch (error) {
      console.error("Preview error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to generate preview",
      });
    }
  });

  // Test one chunk endpoint
  app.post("/api/test-chunk", async (req, res) => {
    try {
      const { text, config } = req.body as {
        text: string;
        config: ProcessingConfig;
      };

      if (!text || !config) {
        return res.status(400).json({ error: "Missing text or configuration" });
      }

      // Process just the first chunk (up to batchSize sentences)
      const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
      const testChunk = sentences.slice(0, config.batchSize).join(" ");

      const processedText = await llmService.processChunk({
        text: testChunk,
        cleaningOptions: config.cleaningOptions,
        speakerConfig: config.speakerConfig,
        modelName: config.modelName,
        customInstructions: config.customInstructions,
      });

      res.json({
        originalChunk: testChunk,
        processedChunk: processedText,
        sentenceCount: Math.min(sentences.length, config.batchSize),
      });
    } catch (error) {
      console.error("Test chunk error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to test chunk",
      });
    }
  });

  // File upload endpoint
  app.post("/api/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      let text = "";
      const fileType = req.file.originalname.toLowerCase();

      if (fileType.endsWith(".txt")) {
        text = req.file.buffer.toString("utf-8");
      } else if (fileType.endsWith(".epub")) {
        text = await parseEpub(req.file.buffer);
      } else {
        return res.status(400).json({ error: "Unsupported file type" });
      }

      // Calculate stats
      const wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;
      const charCount = text.length;

      res.json({
        fileName: req.file.originalname,
        fileType: req.file.mimetype,
        text,
        wordCount,
        charCount,
      });
    } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Failed to process file",
      });
    }
  });

  // WebSocket server for real-time processing
  const wss = new WebSocketServer({ noServer: true });

  wss.on("connection", (ws: WebSocket) => {
    ws.on("message", async (message: Buffer) => {
      try {
        const data = JSON.parse(message.toString());
        const { text, config } = data as {
          text: string;
          config: ProcessingConfig;
        };

        if (!text || !config) {
          ws.send(
            JSON.stringify({
              type: "error",
              payload: {
                message: "Missing text or configuration",
              },
            } as WSMessage)
          );
          return;
        }

        // Send log entry
        const startLog: LogEntry = {
          id: nanoid(),
          timestamp: new Date(),
          type: "info",
          message: `Starting text processing with ${config.modelName}`,
          details: `Batch size: ${config.batchSize} sentences`,
        };

        ws.send(
          JSON.stringify({
            type: "log",
            payload: startLog,
          } as WSMessage)
        );

        // Split text to get total chunks
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        const totalChunks = Math.ceil(sentences.length / config.batchSize);

        // Process text with progress updates
        const processedText = await textProcessor.processText(
          text,
          config,
          (progress) => {
            // Send progress update
            ws.send(
              JSON.stringify({
                type: "progress",
                payload: {
                  progress: ((progress.chunkIndex + 1) / totalChunks) * 100,
                  currentChunk: progress.chunkIndex + 1,
                  totalChunks,
                },
              } as WSMessage)
            );

            // Send chunk result
            ws.send(
              JSON.stringify({
                type: "chunk",
                payload: progress,
              } as WSMessage)
            );

            // Send log for retries and failures
            if (progress.status === "retry") {
              const retryLog: LogEntry = {
                id: nanoid(),
                timestamp: new Date(),
                type: "warning",
                message: `Chunk ${progress.chunkIndex + 1} validation failed, retrying...`,
                details: `Retry attempt ${progress.retryCount}`,
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: retryLog,
                } as WSMessage)
              );
            } else if (progress.status === "failed") {
              const failLog: LogEntry = {
                id: nanoid(),
                timestamp: new Date(),
                type: "error",
                message: `Chunk ${progress.chunkIndex + 1} failed after retries`,
                details: "Using original text for this chunk",
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: failLog,
                } as WSMessage)
              );
            } else if (progress.status === "success") {
              const successLog: LogEntry = {
                id: nanoid(),
                timestamp: new Date(),
                type: "success",
                message: `Chunk ${progress.chunkIndex + 1} processed successfully`,
              };

              ws.send(
                JSON.stringify({
                  type: "log",
                  payload: successLog,
                } as WSMessage)
              );
            }
          }
        );

        // Send completion
        ws.send(
          JSON.stringify({
            type: "complete",
            payload: {
              processedText,
              totalChunks,
            },
          } as WSMessage)
        );
      } catch (error) {
        console.error("Processing error:", error);
        ws.send(
          JSON.stringify({
            type: "error",
            payload: {
              message: error instanceof Error ? error.message : "Processing failed",
              details: error instanceof Error ? error.stack : undefined,
            },
          } as WSMessage)
        );
      }
    });

    ws.on("error", (error) => {
      console.error("WebSocket error:", error);
    });
  });

  // Handle WebSocket upgrade
  httpServer.on("upgrade", (request, socket, head) => {
    if (request.url === "/ws/process") {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit("connection", ws, request);
      });
    } else {
      socket.destroy();
    }
  });

  return httpServer;
}
