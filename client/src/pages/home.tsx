import { useState, useEffect, useRef, useCallback } from "react";
import { FileUpload } from "@/components/file-upload";
import { CleaningOptionsPanel } from "@/components/cleaning-options";
import { SpeakerConfigPanel } from "@/components/speaker-config";
import { CharacterExtraction } from "@/components/character-extraction";
import { ModelSourceSelector } from "@/components/model-source-selector";
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

const ensureFixHyphenation = (options: CleaningOptions): CleaningOptions => ({
  ...options,
  fixHyphenation: options.fixHyphenation ?? false,
});

const ensureNarratorDefaults = (config: SpeakerConfig): SpeakerConfig => ({
  ...config,
  narratorAttribution: config.narratorAttribution ?? "remove",
  characterMapping: config.characterMapping ?? [],
});

export default function Home() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileStats, setFileStats] = useState<{
    wordCount: number;
    charCount: number;
  } | null>(null);
  const [originalText, setOriginalText] = useState("");
  
  const [cleaningOptions, setCleaningOptions] = useState<CleaningOptions>(() =>
    ensureFixHyphenation({
      replaceSmartQuotes: true,
      fixOcrErrors: true,
      correctSpelling: false,
      removeUrls: true,
      removeFootnotes: true,
      addPunctuation: true,
      fixHyphenation: false,
    })
  );

  const updateCleaningOptions = useCallback(
    (updater: CleaningOptions | ((prev: CleaningOptions) => CleaningOptions)) => {
      setCleaningOptions((prev) => {
        const next = typeof updater === "function" ? updater(prev) : updater;
        return ensureFixHyphenation(next);
      });
    },
    []
  );

  const [speakerConfig, setSpeakerConfig] = useState<SpeakerConfig>(() =>
    ensureNarratorDefaults({
      mode: "format",
      speakerCount: 2,
      labelFormat: "speaker",
      extractCharacters: false,
      sampleSize: 50,
      includeNarrator: false,
      narratorAttribution: "remove",
      characterMapping: [],
    })
  );

  const updateSpeakerConfig = useCallback(
    (updater: SpeakerConfig | ((prev: SpeakerConfig) => SpeakerConfig)) => {
      setSpeakerConfig((prev) => {
        const next = typeof updater === "function" ? updater(prev) : updater;
        return ensureNarratorDefaults(next);
      });
    },
    []
  );

  const [batchSize, setBatchSize] = useState(10);
  const [modelSource, setModelSource] = useState<"api" | "ollama">("api");
  const [modelName, setModelName] = useState("meta-llama/Llama-3.1-8B-Instruct");
  const [ollamaModelName, setOllamaModelName] = useState<string>();
  const [customInstructions, setCustomInstructions] = useState("");
  const [singlePass, setSinglePass] = useState(false);
  const [concisePrompts, setConcisePrompts] = useState(false);
  const [extendedExamples, setExtendedExamples] = useState(false);

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentChunk, setCurrentChunk] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [etaMs, setEtaMs] = useState<number | undefined>(undefined);
  const [lastChunkMs, setLastChunkMs] = useState<number | undefined>(undefined);
  const [avgChunkMs, setAvgChunkMs] = useState<number | undefined>(undefined);
  const [totalInputTokens, setTotalInputTokens] = useState<number | undefined>(undefined);
  const [totalOutputTokens, setTotalOutputTokens] = useState<number | undefined>(undefined);
  const [totalCost, setTotalCost] = useState<number | undefined>(undefined);
  const [estimatedTotalCost, setEstimatedTotalCost] = useState<number | undefined>(undefined);
  const [processedText, setProcessedText] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isTesting, setIsTesting] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  // Auto-select first API model from good_models.json and default batch size
  useEffect(() => {
    if (modelSource !== 'api') return;
    (async () => {
      try {
        const res = await fetch('/api/good-models');
        if (!res.ok) return;
        const data = await res.json();
        const models = Array.isArray(data?.models) ? data.models : [];
        const first = models[0];
        if (!first) return;
        const id = typeof first === 'string' ? first : first.id;
        const rec = typeof first === 'string' ? undefined : first.recommendedChunkSize;
        // Set default only if still on initial default
        if (modelName === 'meta-llama/Llama-3.1-8B-Instruct') {
          setModelName(id);
        }
        if (typeof rec === 'number') {
          setBatchSize(rec);
        }
      } catch {}
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/process`);
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
            modelSource,
            modelName,
            ollamaModelName,
            customInstructions: customInstructions || undefined,
            singlePass,
            concisePrompts,
            extendedExamples,
          },
        })
      );

      const modelInfo = modelSource === 'ollama' 
        ? `Ollama: ${ollamaModelName || 'default'}` 
        : `API: ${modelName}`;
      addLog("info", "Processing started", `Model: ${modelInfo}, Batch size: ${batchSize}`);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        switch (message.type) {
          case "progress":
            setProgress(message.payload.progress);
            setCurrentChunk(message.payload.currentChunk);
            setTotalChunks(message.payload.totalChunks);
            if (typeof message.payload.etaMs === 'number') setEtaMs(message.payload.etaMs);
            if (typeof message.payload.lastChunkMs === 'number') setLastChunkMs(message.payload.lastChunkMs);
            if (typeof message.payload.avgChunkMs === 'number') setAvgChunkMs(message.payload.avgChunkMs);
            if (typeof message.payload.totalInputTokens === 'number') setTotalInputTokens(message.payload.totalInputTokens);
            if (typeof message.payload.totalOutputTokens === 'number') setTotalOutputTokens(message.payload.totalOutputTokens);
            if (typeof message.payload.totalCost === 'number') {
              setTotalCost(message.payload.totalCost);
              const cc = message.payload.currentChunk;
              const tc = message.payload.totalChunks;
              if (typeof cc === 'number' && cc > 0 && typeof tc === 'number' && tc > 0) {
                setEstimatedTotalCost((message.payload.totalCost / cc) * tc);
              }
            }
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

          case "log": {
            const payload = message.payload || {};
            const ts = payload.timestamp;
            const timestamp = typeof ts === 'string' ? new Date(ts) : (ts instanceof Date ? ts : new Date());
            setLogs((prev) => [...prev, { ...payload, timestamp }]);
            break;
          }

          case "complete":
            setIsProcessing(false);
            setProcessedText(message.payload.processedText);
            setEtaMs(undefined);
            setLastChunkMs(undefined);
            setAvgChunkMs(undefined);
            if (typeof message.payload.totalInputTokens === 'number') setTotalInputTokens(message.payload.totalInputTokens);
            if (typeof message.payload.totalOutputTokens === 'number') setTotalOutputTokens(message.payload.totalOutputTokens);
            if (typeof message.payload.totalCost === 'number') {
              setTotalCost(message.payload.totalCost);
              setEstimatedTotalCost(message.payload.totalCost);
            }
            addLog(
              "success",
              "Processing completed",
              `${message.payload.totalChunks} chunks processed successfully` +
              (typeof totalCost === 'number' ? ` — total cost: $${totalCost.toFixed(4)}` : '')
            );
            toast({
              title: "Processing complete",
              description: typeof totalCost === 'number' ? `Text processed. Estimated cost $${totalCost.toFixed(4)}` : "Text has been processed successfully",
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
            modelSource,
            modelName,
            ollamaModelName,
            customInstructions: customInstructions || undefined,
            singlePass,
            concisePrompts,
            extendedExamples,
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to test chunk");
      }

          const data = await response.json();
          // Estimate total cost based on test chunk usage and total chunks
          if (data && data.usage) {
            try {
              const chunkCost = (data.usage.inputCost || 0) + (data.usage.outputCost || 0);
              const sentences = originalText.match(/[^.!?]+(?:[.!?]+|$)/g) || [originalText];
              const totalChunksEstimate = Math.ceil(sentences.length / batchSize);
              setEstimatedTotalCost(chunkCost * totalChunksEstimate);
            } catch {}
          }
          setProcessedText(
        `=== TEST RESULT (${data.sentenceCount} sentences) ===\n\nOriginal:\n${data.originalChunk}\n\n---\n\nProcessed:\n${data.processedChunk}` +
        (data.usage ? `\n\n---\nUsage: in ${data.usage.inputTokens} tok, out ${data.usage.outputTokens} tok — cost: $${(((data.usage.inputCost||0)+(data.usage.outputCost||0)).toFixed(4))}` : '')
          );

          addLog(
            "success",
            "Test completed",
            `Processed ${data.sentenceCount} sentences` + (data.usage ? ` — cost: $${(((data.usage.inputCost||0)+(data.usage.outputCost||0)).toFixed(4))}` : '')
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
      <div className="grid grid-cols-1 lg:grid-cols-[480px_1fr_384px] h-screen">
        {/* Left Panel - Configuration */}
        <div className="border-r bg-card overflow-y-auto">
          <div className="p-4 pb-20 space-y-3">
            <div>
              <h2 className="text-lg font-semibold mb-3">Configuration</h2>
              <div className="space-y-3">
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
              onChange={updateCleaningOptions}
              disabled={isProcessing}
            />

            <SpeakerConfigPanel
              config={speakerConfig}
              onChange={updateSpeakerConfig}
              disabled={isProcessing}
            />

            {speakerConfig.mode === "intelligent" && (
              <CharacterExtraction
                text={originalText}
                modelSource={modelSource}
                modelName={modelName}
                ollamaModelName={ollamaModelName}
                characterMapping={speakerConfig.characterMapping}
                sampleSize={speakerConfig.sampleSize}
                includeNarrator={speakerConfig.includeNarrator}
                  onSampleSizeChange={(size) =>
                    updateSpeakerConfig((prev) => ({ ...prev, sampleSize: size }))
                  }
                  onIncludeNarratorChange={(include) =>
                    updateSpeakerConfig((prev) => ({ ...prev, includeNarrator: include }))
                  }
                  onCharactersExtracted={(characters) => {
                    updateSpeakerConfig((prev) => ({ ...prev, characterMapping: characters }));
                    addLog(
                      "success",
                      `Characters extracted: ${characters.length} character(s)`,
                      characters.map(c => `${c.name} = Speaker ${c.speakerNumber}`).join(", ")
                    );
                  }}
                  onNarratorCharacterNameChange={(name) => {
                    updateSpeakerConfig((prev) => ({
                      ...prev,
                      narratorCharacterName: name || undefined,
                    }));
                    if (name) {
                      addLog("info", `Narrator identified as: ${name}`);
                    }
                  }}
                disabled={isProcessing}
              />
            )}

            <ModelSourceSelector
              modelSource={modelSource}
              ollamaModelName={ollamaModelName}
              onModelSourceChange={setModelSource}
              onOllamaModelChange={setOllamaModelName}
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
              singlePass={singlePass}
              concisePrompts={concisePrompts}
              extendedExamples={extendedExamples}
              disabled={isProcessing}
            />

            <ProcessingControls
            batchSize={batchSize}
            onBatchSizeChange={setBatchSize}
            modelName={modelName}
            onModelNameChange={setModelName}
            estimatedTotalCost={estimatedTotalCost}
            singlePass={singlePass}
            onSinglePassChange={setSinglePass}
            concisePrompts={concisePrompts}
            onConcisePromptsChange={setConcisePrompts}
            extendedExamples={extendedExamples}
            onExtendedExamplesChange={setExtendedExamples}
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
          <div className="p-4 pb-3">
            <ProgressDisplay
              progress={progress}
              currentChunk={currentChunk}
              totalChunks={totalChunks}
              isProcessing={isProcessing}
              etaMs={etaMs}
              lastChunkMs={lastChunkMs}
              avgChunkMs={avgChunkMs}
              totalInputTokens={totalInputTokens}
              totalOutputTokens={totalOutputTokens}
              totalCost={totalCost}
            />
          </div>
          <div className="flex-1 px-4 pb-4 overflow-hidden">
            <OutputDisplay text={processedText} fileName={selectedFile?.name} />
          </div>
        </div>

        {/* Right Panel - Activity Log */}
        <div className="border-l bg-card overflow-hidden">
          <div className="h-full p-4">
            <ActivityLog logs={logs} onClear={handleClearLogs} />
          </div>
        </div>
      </div>
    </div>
  );
}
