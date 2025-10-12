import { useState, useEffect, useRef } from "react";
import { FileUpload } from "@/components/file-upload";
import { CleaningOptionsPanel } from "@/components/cleaning-options";
import { SpeakerConfigPanel } from "@/components/speaker-config";
import { ProcessingControls } from "@/components/processing-controls";
import { CustomInstructions } from "@/components/custom-instructions";
import { PromptPreview } from "@/components/prompt-preview";
import { ProgressDisplay } from "@/components/progress-display";
import { OutputDisplay } from "@/components/output-display";
import { ActivityLog } from "@/components/activity-log";
import { useToast } from "@/hooks/use-toast";
import { nanoid } from "nanoid";
import type {
  CleaningOptions,
  SpeakerConfig,
  LogEntry,
  FileUploadResponse,
} from "@shared/schema";

export default function Home() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileStats, setFileStats] = useState<{
    wordCount: number;
    charCount: number;
  } | null>(null);
  const [originalText, setOriginalText] = useState("");
  
  const [cleaningOptions, setCleaningOptions] = useState<CleaningOptions>({
    replaceSmartQuotes: true,
    fixOcrErrors: true,
    correctSpelling: false,
    removeUrls: true,
    removeFootnotes: true,
    addPunctuation: true,
  });

  const [speakerConfig, setSpeakerConfig] = useState<SpeakerConfig>({
    mode: "format",
    speakerCount: 2,
    labelFormat: "speaker",
  });

  const [batchSize, setBatchSize] = useState(10);
  const [modelName, setModelName] = useState("Qwen/Qwen2.5-72B-Instruct");
  const [customInstructions, setCustomInstructions] = useState("");

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentChunk, setCurrentChunk] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [processedText, setProcessedText] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isTesting, setIsTesting] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  const addLog = (
    type: LogEntry["type"],
    message: string,
    details?: string
  ) => {
    const newLog: LogEntry = {
      id: nanoid(),
      timestamp: new Date(),
      type,
      message,
      details,
    };
    setLogs((prev) => [...prev, newLog]);
  };

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setProcessedText("");
    setProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload file");
      }

      const data: FileUploadResponse = await response.json();
      setFileStats({
        wordCount: data.wordCount,
        charCount: data.charCount,
      });
      setOriginalText(data.text);

      addLog("success", `File uploaded: ${file.name}`, `${data.wordCount} words, ${data.charCount} characters`);
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload file",
        variant: "destructive",
      });
      addLog("error", "File upload failed", error instanceof Error ? error.message : undefined);
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setFileStats(null);
    setOriginalText("");
    setProcessedText("");
    setProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);
  };

  const handleStartProcessing = () => {
    if (!originalText) return;

    setIsProcessing(true);
    setProcessedText("");
    setProgress(0);
    setCurrentChunk(0);

    const ws = new WebSocket(`ws://${window.location.host}/ws/process`);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog("info", "Connected to processing server");
      
      ws.send(
        JSON.stringify({
          text: originalText,
          config: {
            batchSize,
            cleaningOptions,
            speakerConfig,
            modelName,
            customInstructions: customInstructions || undefined,
          },
        })
      );

      addLog("info", "Processing started", `Model: ${modelName}, Batch size: ${batchSize}`);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        switch (message.type) {
          case "progress":
            setProgress(message.payload.progress);
            setCurrentChunk(message.payload.currentChunk);
            setTotalChunks(message.payload.totalChunks);
            break;

          case "chunk":
            setProcessedText((prev) =>
              prev ? `${prev}\n${message.payload.processedText}` : message.payload.processedText
            );
            if (message.payload.status === "retry") {
              addLog(
                "warning",
                `Chunk ${message.payload.chunkIndex} retry attempt ${message.payload.retryCount}`
              );
            } else if (message.payload.status === "failed") {
              addLog(
                "error",
                `Chunk ${message.payload.chunkIndex} failed after retries`
              );
            }
            break;

          case "log":
            setLogs((prev) => [...prev, message.payload]);
            break;

          case "complete":
            setIsProcessing(false);
            setProcessedText(message.payload.processedText);
            addLog(
              "success",
              "Processing completed",
              `${message.payload.totalChunks} chunks processed successfully`
            );
            toast({
              title: "Processing complete",
              description: "Text has been processed successfully",
            });
            break;

          case "error":
            setIsProcessing(false);
            addLog("error", message.payload.message, message.payload.details);
            toast({
              title: "Processing error",
              description: message.payload.message,
              variant: "destructive",
            });
            break;
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onerror = () => {
      setIsProcessing(false);
      addLog("error", "WebSocket connection error");
      toast({
        title: "Connection error",
        description: "Failed to connect to processing server",
        variant: "destructive",
      });
    };

    ws.onclose = () => {
      if (isProcessing) {
        setIsProcessing(false);
        addLog("info", "Connection closed");
      }
    };
  };

  const handleStopProcessing = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsProcessing(false);
    addLog("warning", "Processing stopped by user");
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleTestChunk = async () => {
    if (!originalText) return;

    setIsTesting(true);
    addLog("info", "Testing one chunk...");

    try {
      const response = await fetch("/api/test-chunk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: originalText,
          config: {
            batchSize,
            cleaningOptions,
            speakerConfig,
            modelName,
            customInstructions: customInstructions || undefined,
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to test chunk");
      }

      const data = await response.json();
      setProcessedText(
        `=== TEST RESULT (${data.sentenceCount} sentences) ===\n\nOriginal:\n${data.originalChunk}\n\n---\n\nProcessed:\n${data.processedChunk}`
      );

      addLog(
        "success",
        "Test completed",
        `Processed ${data.sentenceCount} sentences`
      );

      toast({
        title: "Test complete",
        description: "One chunk has been processed successfully",
      });
    } catch (error) {
      addLog(
        "error",
        "Test failed",
        error instanceof Error ? error.message : undefined
      );
      toast({
        title: "Test failed",
        description:
          error instanceof Error ? error.message : "Failed to test chunk",
        variant: "destructive",
      });
    } finally {
      setIsTesting(false);
    }
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr_384px] h-screen">
        {/* Left Panel - Configuration */}
        <div className="border-r bg-card overflow-y-auto">
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4">Configuration</h2>
              <div className="space-y-4">
                <FileUpload
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  onClear={handleClearFile}
                  fileStats={fileStats || undefined}
                  isProcessing={isProcessing}
                />
              </div>
            </div>

            <CleaningOptionsPanel
              options={cleaningOptions}
              onChange={setCleaningOptions}
              disabled={isProcessing}
            />

            <SpeakerConfigPanel
              config={speakerConfig}
              onChange={setSpeakerConfig}
              disabled={isProcessing}
            />

            <CustomInstructions
              value={customInstructions}
              onChange={setCustomInstructions}
              disabled={isProcessing}
            />

            <PromptPreview
              sampleText={originalText}
              cleaningOptions={cleaningOptions}
              speakerConfig={speakerConfig}
              customInstructions={customInstructions}
              disabled={isProcessing}
            />

            <ProcessingControls
              batchSize={batchSize}
              onBatchSizeChange={setBatchSize}
              modelName={modelName}
              onModelNameChange={setModelName}
              onStart={handleStartProcessing}
              onStop={handleStopProcessing}
              onTest={handleTestChunk}
              isProcessing={isProcessing}
              canStart={!!originalText && !isProcessing}
              isTesting={isTesting}
            />
          </div>
        </div>

        {/* Center Panel - Output */}
        <div className="flex flex-col overflow-hidden">
          <div className="p-6 pb-4">
            <ProgressDisplay
              progress={progress}
              currentChunk={currentChunk}
              totalChunks={totalChunks}
              isProcessing={isProcessing}
            />
          </div>
          <div className="flex-1 px-6 pb-6 overflow-hidden">
            <OutputDisplay text={processedText} fileName={selectedFile?.name} />
          </div>
        </div>

        {/* Right Panel - Activity Log */}
        <div className="border-l bg-card overflow-hidden">
          <div className="h-full p-6">
            <ActivityLog logs={logs} onClear={handleClearLogs} />
          </div>
        </div>
      </div>
    </div>
  );
}
