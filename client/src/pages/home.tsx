import { useState, useEffect, useRef, useCallback } from "react";
import { FileUpload } from "@/components/file-upload";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
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
import { useLocation } from "wouter";
import { nanoid } from "nanoid";
import {
  ArrowRight,
  AudioLines,
  BookOpenText,
  CheckCircle2,
  ChevronDown,
  FilePenLine,
  Loader2,
  ScanText,
  Sparkles,
  Wand2,
} from "lucide-react";
import { countWords, segmentSentences } from "@shared/text-utils";
import type {
  CleaningOptions,
  SpeakerConfig,
  LogEntry,
  FileUploadResponse,
  DeterministicCleanResponse,
} from "@shared/schema";

const ensureFixHyphenation = (options: CleaningOptions): CleaningOptions => ({
  ...options,
  insertChapterBreaks: options.insertChapterBreaks ?? false,
  fixHyphenation: options.fixHyphenation ?? false,
});

const ensureNarratorDefaults = (config: SpeakerConfig): SpeakerConfig => ({
  ...config,
  narratorAttribution: config.narratorAttribution ?? "remove",
  characterMapping: config.characterMapping ?? [],
});

interface TestChunkPreview {
  originalChunk: string;
  processedChunk: string;
  sentenceCount: number;
  usage?: {
    inputTokens: number;
    outputTokens: number;
    inputCost?: number;
    outputCost?: number;
  };
}

export default function Home() {
  const { toast } = useToast();
  const [, navigate] = useLocation();
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
      insertChapterBreaks: false,
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
      mode: "none",
      speakerCount: 2,
      labelFormat: "speaker",
      extractCharacters: false,
      sampleSize: 50,
      includeNarrator: true,
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
  const [temperature, setTemperature] = useState<number>(0.3);
  const [llmCleaningDisabled, setLlmCleaningDisabled] = useState<boolean>(false);
  const [customInstructions, setCustomInstructions] = useState("");
  const [singlePass, setSinglePass] = useState(false);
  const [extendedExamples, setExtendedExamples] = useState(false);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isCleaning, setIsCleaning] = useState(false);
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
  const [testPreview, setTestPreview] = useState<TestChunkPreview | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const testAbortRef = useRef<AbortController | null>(null);
  const isProcessingRef = useRef(false);
  const totalCostRef = useRef<number | undefined>(undefined);

  // Load settings from localStorage (persisted between runs)
  useEffect(() => {
    try {
      const raw = localStorage.getItem('vf_settings_v2');
      if (raw) {
        const cfg = JSON.parse(raw);
        if (typeof cfg.batchSize === 'number') setBatchSize(cfg.batchSize);
        if (cfg.cleaningOptions) setCleaningOptions(ensureFixHyphenation(cfg.cleaningOptions));
        if (cfg.speakerConfig) setSpeakerConfig(ensureNarratorDefaults(cfg.speakerConfig));
        if (cfg.modelSource === 'api' || cfg.modelSource === 'ollama') setModelSource(cfg.modelSource);
        if (typeof cfg.modelName === 'string') setModelName(cfg.modelName);
        if (typeof cfg.ollamaModelName === 'string') setOllamaModelName(cfg.ollamaModelName);
        if (typeof cfg.customInstructions === 'string') setCustomInstructions(cfg.customInstructions);
        if (typeof cfg.singlePass === 'boolean') setSinglePass(cfg.singlePass);
        if (typeof cfg.extendedExamples === 'boolean') setExtendedExamples(cfg.extendedExamples);
        if (typeof cfg.temperature === 'number') setTemperature(cfg.temperature);
        if (typeof cfg.llmCleaningDisabled === 'boolean') setLlmCleaningDisabled(cfg.llmCleaningDisabled);
      }
    } catch {}
  }, []);

  // Auto-select first API model from good_models.json and default batch size (only if not set)
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
        // Set default only if still on initial default and no saved preference
        if (!localStorage.getItem('vf_settings_v2') && modelName === 'meta-llama/Llama-3.1-8B-Instruct') {
          setModelName(id);
        }
        if (typeof rec === 'number') {
          if (!localStorage.getItem('vf_settings_v2')) setBatchSize(rec);
        }
      } catch {}
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist settings to localStorage on change
  useEffect(() => {
    try {
      const cfg = {
        batchSize,
        cleaningOptions,
        speakerConfig,
        modelSource,
        modelName,
        ollamaModelName,
        customInstructions,
        singlePass,
        extendedExamples,
        temperature,
        llmCleaningDisabled,
      };
      localStorage.setItem('vf_settings_v2', JSON.stringify(cfg));
    } catch {}
  }, [batchSize, cleaningOptions, speakerConfig, modelSource, modelName, ollamaModelName, customInstructions, singlePass, extendedExamples, temperature, llmCleaningDisabled]);

  // Recover the current workspace after a refresh, or accept a handoff from OCR.
  useEffect(() => {
    try {
      const incoming = sessionStorage.getItem("vf_prepare_draft");
      const saved = localStorage.getItem("vf_project_draft_v2");
      const draft = incoming ? { source: incoming, output: "" } : saved ? JSON.parse(saved) : null;
      if (draft && typeof draft.source === "string" && draft.source.trim()) {
        setOriginalText(draft.source);
        setProcessedText(typeof draft.output === "string" ? draft.output : "");
        setFileStats({ wordCount: countWords(draft.source), charCount: draft.source.length });
      }
      if (incoming) sessionStorage.removeItem("vf_prepare_draft");
    } catch {
      // A malformed or oversized browser draft should never block the editor.
    }
  }, []);

  useEffect(() => {
    if (!originalText.trim() || originalText.length > 750_000) {
      try {
        localStorage.removeItem("vf_project_draft_v2");
      } catch {}
      return;
    }
    try {
      localStorage.setItem(
        "vf_project_draft_v2",
        JSON.stringify({ source: originalText, output: processedText, updatedAt: Date.now() })
      );
    } catch {
      // Storage can be unavailable in private browsing or full for long books.
    }
  }, [originalText, processedText]);

  // When switching to Ollama, attempt to select the first installed model
  useEffect(() => {
    if (modelSource !== 'ollama') return;
    if (ollamaModelName && ollamaModelName.trim().length > 0) return;
    let mounted = true;
    (async () => {
      try {
        const res = await fetch('/api/ollama/models');
        if (!res.ok) return;
        const data = await res.json();
        const models: string[] = Array.isArray(data?.models)
          ? data.models.map((m: any) => (typeof m === 'string' ? m : m?.id)).filter(Boolean)
          : [];
        if (mounted && models.length > 0) {
          setOllamaModelName(models[0]);
        }
      } catch {}
    })();
    return () => { mounted = false; };
  }, [modelSource, ollamaModelName]);

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
    setLogs((prev) => [...prev, newLog].slice(-2_000));
  };

  const handleFileSelect = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.error || "Failed to upload file");
      }

      const data: FileUploadResponse = await response.json();
      setSelectedFile(file);
      setFileStats({
        wordCount: data.wordCount,
        charCount: data.charCount,
      });
      setOriginalText(data.text);
      setProcessedText("");
      setTestPreview(null);
      setProgress(0);
      setCurrentChunk(0);
      setTotalChunks(0);

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
    setTestPreview(null);
    setProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);
  };

  const handleSourceChange = (value: string) => {
    setOriginalText(value);
    setFileStats(value ? { wordCount: countWords(value), charCount: value.length } : null);
    setProcessedText("");
    setTestPreview(null);
    setProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);
  };

  const loadSample = () => {
    const sample = `CHAPTER ONE\n\nThe platform was nearly empty when Mara arrived. Rain ticked against the glass roof.\n\n“Jonas?” she called. “Are you here?”\n\nA figure stepped from behind the timetable. “You came,” Jonas said, lowering his hood.\n\n“I said I would.” Mara glanced toward the tracks. “Now tell me why the last train never arrived.”`;
    setSelectedFile(null);
    handleSourceChange(sample);
    addLog("info", "Loaded the guided sample");
  };

  const applyPreset = (preset: "narration" | "dialogue" | "ocr") => {
    if (preset === "narration") {
      updateSpeakerConfig((previous) => ({ ...previous, mode: "none", includeNarrator: true }));
      updateCleaningOptions((previous) => ({ ...previous, fixHyphenation: false, correctSpelling: false }));
      setSinglePass(false);
    } else if (preset === "dialogue") {
      updateSpeakerConfig((previous) => ({
        ...previous,
        mode: "intelligent",
        includeNarrator: true,
        narratorAttribution: "contextual",
      }));
      setSinglePass(true);
    } else {
      updateSpeakerConfig((previous) => ({ ...previous, mode: "none", includeNarrator: true }));
      updateCleaningOptions({
        replaceSmartQuotes: true,
        fixOcrErrors: true,
        correctSpelling: false,
        removeUrls: true,
        removeFootnotes: true,
        addPunctuation: false,
        insertChapterBreaks: false,
        fixHyphenation: true,
      });
      setSinglePass(false);
    }
    toast({ title: "Recipe applied", description: `${preset === "ocr" ? "OCR repair" : preset === "dialogue" ? "Dialogue cast" : "Clean narration"} settings are ready.` });
  };

  const sendToVoiceStudio = () => {
    const draft = processedText.trim() || originalText.trim();
    if (!draft) return;
    sessionStorage.setItem("vf_tts_draft", draft);
    navigate("/tts");
  };

  const handleStartProcessing = () => {
    if (!originalText) return;
    if (modelSource === "ollama" && !ollamaModelName?.trim()) {
      toast({
        title: "Choose a local model",
        description: "Start Ollama and enter the name of an installed model before processing.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    isProcessingRef.current = true;
    totalCostRef.current = undefined;
    setTestPreview(null);
    setProgress(0);
    setCurrentChunk(0);

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/process`);
    wsRef.current = ws;
    const isCurrentSocket = () => wsRef.current === ws;
    const closeCurrentSocket = () => {
      if (!isCurrentSocket()) return;
      wsRef.current = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };

    ws.onopen = () => {
      if (!isCurrentSocket()) {
        ws.close();
        return;
      }
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
            temperature,
            llmCleaningDisabled,
            customInstructions: customInstructions || undefined,
            singlePass,
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
      if (!isCurrentSocket()) return;
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
              totalCostRef.current = message.payload.totalCost;
              setTotalCost(message.payload.totalCost);
              const cc = message.payload.currentChunk;
              const tc = message.payload.totalChunks;
              if (typeof cc === 'number' && cc > 0 && typeof tc === 'number' && tc > 0) {
                setEstimatedTotalCost((message.payload.totalCost / cc) * tc);
              }
            }
            break;

          case "chunk":
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
            setLogs((prev) => [...prev, { ...payload, timestamp }].slice(-2_000));
            break;
          }

          case "complete":
            closeCurrentSocket();
            setIsProcessing(false);
            isProcessingRef.current = false;
            setProcessedText(message.payload.processedText);
            setEtaMs(undefined);
            setLastChunkMs(undefined);
            setAvgChunkMs(undefined);
            if (typeof message.payload.totalInputTokens === 'number') setTotalInputTokens(message.payload.totalInputTokens);
            if (typeof message.payload.totalOutputTokens === 'number') setTotalOutputTokens(message.payload.totalOutputTokens);
            if (typeof message.payload.totalCost === 'number') {
              totalCostRef.current = message.payload.totalCost;
              setTotalCost(message.payload.totalCost);
              setEstimatedTotalCost(message.payload.totalCost);
            }
            const finalCost = typeof message.payload.totalCost === "number"
              ? message.payload.totalCost
              : totalCostRef.current;
            const failedChunks = typeof message.payload.failedChunks === "number" ? message.payload.failedChunks : 0;
            addLog(
              failedChunks > 0 ? "warning" : "success",
              failedChunks > 0 ? "Processing completed with warnings" : "Processing completed",
              (failedChunks > 0
                ? `${failedChunks} of ${message.payload.totalChunks} chunks kept their original text after retry failures`
                : `${message.payload.totalChunks} chunks passed validation`) +
              (typeof finalCost === 'number' ? ` — estimated cost: $${finalCost.toFixed(4)}` : '')
            );
            toast({
              title: failedChunks > 0 ? "Review required" : "Processing complete",
              description: failedChunks > 0
                ? `${failedChunks} chunk${failedChunks === 1 ? "" : "s"} fell back to the original text. Check Run details before export.`
                : typeof finalCost === 'number' ? `Text processed. Estimated cost $${finalCost.toFixed(4)}` : "Text has been processed successfully",
              variant: failedChunks > 0 ? "destructive" : undefined,
            });
            break;

          case "error":
            closeCurrentSocket();
            setIsProcessing(false);
            isProcessingRef.current = false;
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
      if (!isCurrentSocket()) return;
      closeCurrentSocket();
      setIsProcessing(false);
      isProcessingRef.current = false;
      addLog("error", "WebSocket connection error");
      toast({
        title: "Connection error",
        description: "Failed to connect to processing server",
        variant: "destructive",
      });
    };

    ws.onclose = () => {
      if (!isCurrentSocket()) return;
      wsRef.current = null;
      if (isProcessingRef.current) {
        setIsProcessing(false);
        isProcessingRef.current = false;
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
    isProcessingRef.current = false;
    addLog("warning", "Processing stopped by user");
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleDeterministicClean = async () => {
    if (!originalText.trim()) {
      toast({
        title: "No text to clean",
        description: "Upload or paste text before running deterministic cleaning.",
        variant: "destructive",
      });
      addLog("warning", "Deterministic cleaning skipped", "No text available");
      return;
    }

    setIsCleaning(true);
    addLog("info", "Running deterministic cleaning (no LLM)...");

    try {
      const response = await fetch("/api/text/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: originalText,
          options: cleaningOptions,
        }),
      });

      let payload: unknown = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok || !payload || typeof payload !== "object" || payload === null) {
        const message =
          payload && typeof payload === "object" && "error" in payload && typeof (payload as any).error === "string"
            ? (payload as any).error
            : `Failed to clean text (${response.status})`;
        throw new Error(message);
      }

      const data = payload as DeterministicCleanResponse;
      const cleaned = data.cleanedText ?? "";
      const applied = Array.isArray(data.appliedSteps) ? data.appliedSteps.filter((step) => typeof step === "string" && step.length > 0) : [];

      setProcessedText(cleaned);
      setProgress(0);
      setCurrentChunk(0);
      setTotalChunks(0);
      setEtaMs(undefined);
      setLastChunkMs(undefined);
      setAvgChunkMs(undefined);
      setTotalInputTokens(undefined);
      setTotalOutputTokens(undefined);
      setTotalCost(undefined);
      setEstimatedTotalCost(undefined);

      const appliedSummary = applied.length > 0 ? `Applied: ${applied.join(", ")}` : undefined;
      addLog("success", "Deterministic cleaning applied", appliedSummary);
      toast({
        title: "Clean version created",
        description: appliedSummary ?? "No deterministic changes were needed. Your source remains untouched.",
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to clean text";
      addLog("error", "Deterministic cleaning failed", message);
      toast({
        title: "Cleaning failed",
        description: message,
        variant: "destructive",
      });
    } finally {
      setIsCleaning(false);
    }
  };

  const handleTestChunk = async () => {
    if (!originalText) return;

    testAbortRef.current?.abort();
    const controller = new AbortController();
    testAbortRef.current = controller;
    setIsTesting(true);
    setTestPreview(null);
    addLog("info", "Testing one chunk...");

    try {
      const response = await fetch("/api/test-chunk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          text: originalText,
          config: {
            batchSize,
            cleaningOptions,
            speakerConfig,
            modelSource,
            modelName,
            ollamaModelName,
            temperature,
            llmCleaningDisabled,
            customInstructions: customInstructions || undefined,
            singlePass,
            extendedExamples,
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to test chunk");
      }

      const data = (await response.json()) as TestChunkPreview;
      if (controller.signal.aborted || testAbortRef.current !== controller) return;
      // Estimate total cost based on test chunk usage and total chunks.
      if (data.usage) {
        const chunkCost = (data.usage.inputCost || 0) + (data.usage.outputCost || 0);
        const sentences = segmentSentences(originalText);
        const totalChunksEstimate = Math.ceil(sentences.length / batchSize);
        setEstimatedTotalCost(chunkCost * totalChunksEstimate);
      }
      setTestPreview(data);

      addLog(
        "success",
        "Test completed",
        `Processed ${data.sentenceCount} sentences` +
          (data.usage
            ? ` — estimated cost: $${((data.usage.inputCost || 0) + (data.usage.outputCost || 0)).toFixed(4)}`
            : "")
      );

      toast({
        title: "Test complete",
        description: "One chunk has been processed successfully",
      });
    } catch (error) {
      if (controller.signal.aborted) return;
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
      if (testAbortRef.current === controller) {
        testAbortRef.current = null;
        setIsTesting(false);
      }
    }
  };

  useEffect(() => {
    const activeRequest = testAbortRef.current;
    if (activeRequest) {
      activeRequest.abort();
      testAbortRef.current = null;
      setIsTesting(false);
    }
    setTestPreview(null);
  }, [
    batchSize,
    cleaningOptions,
    customInstructions,
    extendedExamples,
    llmCleaningDisabled,
    modelName,
    modelSource,
    ollamaModelName,
    originalText,
    singlePass,
    speakerConfig,
    temperature,
  ]);

  useEffect(() => {
    return () => {
      testAbortRef.current?.abort();
      testAbortRef.current = null;
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const sourceWordCount = fileStats?.wordCount ?? countWords(originalText);
  const sourceSentenceCount = originalText ? segmentSentences(originalText).length : 0;
  const hasOutput = processedText.trim().length > 0;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1560px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <header className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <Badge variant="secondary" className="mb-3 rounded-full px-3 py-1 text-[11px] font-bold uppercase tracking-[0.15em]">
              <Sparkles className="mr-1.5 h-3.5 w-3.5 text-primary" />
              Text preparation
            </Badge>
            <h1 className="text-3xl font-bold tracking-[-0.04em] sm:text-4xl">Shape words into a speech-ready script.</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
              Import or paste a draft, apply only the repairs you want, then review every result before it moves to a voice engine.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5 rounded-full border bg-card px-3 py-1.5 shadow-sm">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> Source preserved
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border bg-card px-3 py-1.5 shadow-sm">
              <FilePenLine className="h-3.5 w-3.5 text-primary" />
              {originalText.length > 750_000 ? "Large draft — export to save" : "Local draft autosave"}
            </span>
          </div>
        </header>

        <div className="mt-7 grid gap-3 sm:grid-cols-3">
          <button onClick={() => applyPreset("narration")} className="group rounded-2xl border bg-card p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-md">
            <span className="flex items-center gap-2 text-sm font-bold"><BookOpenText className="h-4 w-4 text-primary" /> Clean narration</span>
            <span className="mt-1 block text-xs leading-5 text-muted-foreground">Preserve prose for a single natural voice.</span>
          </button>
          <button onClick={() => applyPreset("dialogue")} className="group rounded-2xl border bg-card p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-md">
            <span className="flex items-center gap-2 text-sm font-bold"><AudioLines className="h-4 w-4 text-primary" /> Cast dialogue</span>
            <span className="mt-1 block text-xs leading-5 text-muted-foreground">Detect speakers while keeping narration intact.</span>
          </button>
          <button onClick={() => applyPreset("ocr")} className="group rounded-2xl border bg-card p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-md">
            <span className="flex items-center gap-2 text-sm font-bold"><ScanText className="h-4 w-4 text-primary" /> Repair OCR text</span>
            <span className="mt-1 block text-xs leading-5 text-muted-foreground">Fix line-break hyphens, spacing, and scan debris.</span>
          </button>
        </div>

        <div className="mt-5 grid items-start gap-5 xl:grid-cols-[420px_minmax(0,1fr)]">
          <section className="space-y-4" aria-label="Preparation settings">
            <Card className="overflow-hidden rounded-2xl border-card-border shadow-sm">
              <CardContent className="space-y-4 p-4 sm:p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-bold">1. Add your source</p>
                    <p className="mt-1 text-xs leading-5 text-muted-foreground">TXT and EPUB stay local. You can also paste or edit below.</p>
                  </div>
                  {originalText && <Badge variant="outline" className="shrink-0 rounded-full">{sourceWordCount.toLocaleString()} words</Badge>}
                </div>
                <FileUpload
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  onClear={handleClearFile}
                  fileStats={fileStats || undefined}
                  isProcessing={isProcessing || isCleaning}
                />
                <div className="flex items-center gap-3 text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">
                  <span className="h-px flex-1 bg-border" /> or paste text <span className="h-px flex-1 bg-border" />
                </div>
                <Textarea
                  value={originalText}
                  onChange={(event) => handleSourceChange(event.target.value)}
                  disabled={isProcessing || isCleaning}
                  placeholder="Paste a chapter, screenplay, transcript, or rough OCR text…"
                  className="min-h-[190px] resize-y rounded-xl bg-background/60 text-sm leading-6"
                  data-testid="textarea-source"
                />
                <div className="flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
                  <span>{originalText ? `${sourceSentenceCount.toLocaleString()} sentences · ${originalText.length.toLocaleString()} characters` : "Nothing added yet"}</span>
                  {!originalText && <Button variant="ghost" size="sm" onClick={loadSample} className="h-7 px-2 text-[11px] text-primary">Try a sample</Button>}
                </div>
              </CardContent>
            </Card>

            <CleaningOptionsPanel
              options={cleaningOptions}
              onChange={updateCleaningOptions}
              disabled={isProcessing || isCleaning}
            />
            <Button
              onClick={handleDeterministicClean}
              disabled={!originalText || isProcessing || isCleaning}
              variant="secondary"
              className="h-11 w-full rounded-xl"
              data-testid="button-clean-deterministic"
            >
              {isCleaning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Wand2 className="mr-2 h-4 w-4" />}
              {isCleaning ? "Creating clean version…" : "Preview safe cleanup"}
            </Button>
            <p className="-mt-2 px-2 text-[11px] leading-5 text-muted-foreground">
              Runs locally and writes to the result pane. Spelling and added punctuation are AI-assisted and apply during a full run.
            </p>

            <SpeakerConfigPanel
              config={speakerConfig}
              onChange={updateSpeakerConfig}
              disabled={isProcessing || isCleaning}
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
                onSampleSizeChange={(size) => updateSpeakerConfig((prev) => ({ ...prev, sampleSize: size }))}
                onIncludeNarratorChange={(include) => updateSpeakerConfig((prev) => ({ ...prev, includeNarrator: include }))}
                onCharactersExtracted={(characters) => {
                  const highestSpeaker = Math.max(1, ...characters.map((character) => character.speakerNumber));
                  updateSpeakerConfig((prev) => ({
                    ...prev,
                    characterMapping: characters,
                    speakerCount: Math.max(prev.speakerCount, highestSpeaker),
                  }));
                  addLog("success", `Characters extracted: ${characters.length}`, characters.map((character) => `${character.name} = Speaker ${character.speakerNumber}`).join(", "));
                }}
                onNarratorCharacterNameChange={(name) => {
                  updateSpeakerConfig((prev) => ({ ...prev, narratorCharacterName: name || undefined }));
                  if (name) addLog("info", `Narrator identified as: ${name}`);
                }}
                disabled={isProcessing || isCleaning}
              />
            )}

            <details className="group overflow-hidden rounded-2xl border bg-card shadow-sm">
              <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-4 text-sm font-bold marker:hidden">
                AI engine & prompt settings
                <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
              </summary>
              <div className="space-y-3 border-t bg-muted/20 p-3">
                <ModelSourceSelector
                  modelSource={modelSource}
                  ollamaModelName={ollamaModelName}
                  onModelSourceChange={setModelSource}
                  onOllamaModelChange={setOllamaModelName}
                  temperature={temperature}
                  onTemperatureChange={setTemperature}
                  disabled={isProcessing || isCleaning}
                />
                <CustomInstructions value={customInstructions} onChange={setCustomInstructions} disabled={isProcessing || isCleaning} />
                <PromptPreview
                  sampleText={originalText}
                  cleaningOptions={cleaningOptions}
                  speakerConfig={speakerConfig}
                  customInstructions={customInstructions}
                  singlePass={singlePass}
                  llmCleaningDisabled={llmCleaningDisabled}
                  extendedExamples={extendedExamples}
                  disabled={isProcessing || isCleaning}
                />
              </div>
            </details>

            <ProcessingControls
              batchSize={batchSize}
              onBatchSizeChange={setBatchSize}
              modelName={modelName}
              onModelNameChange={setModelName}
              llmCleaningDisabled={llmCleaningDisabled}
              onLlmCleaningDisabledChange={setLlmCleaningDisabled}
              estimatedTotalCost={estimatedTotalCost}
              singlePass={singlePass}
              onSinglePassChange={setSinglePass}
              extendedExamples={extendedExamples}
              onExtendedExamplesChange={setExtendedExamples}
              onStart={handleStartProcessing}
              onStop={handleStopProcessing}
              onTest={handleTestChunk}
              isProcessing={isProcessing}
              canStart={!!originalText.trim() && (modelSource === "api" || !!ollamaModelName?.trim()) && !isProcessing && !isCleaning}
              isTesting={isTesting}
            />
          </section>

          <section className="min-w-0 space-y-4" aria-label="Review result">
            <Card className="rounded-2xl border-card-border shadow-sm">
              <CardContent className="p-4 sm:p-5">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`h-2.5 w-2.5 rounded-full ${isProcessing ? "animate-pulse bg-amber-400" : hasOutput ? "bg-emerald-500" : "bg-muted-foreground/30"}`} />
                      <h2 className="text-sm font-bold">{isProcessing ? "Processing your script" : hasOutput ? "Result ready to review" : "Review workspace"}</h2>
                    </div>
                    <p className="mt-1 pl-[18px] text-xs text-muted-foreground">
                      {hasOutput ? "Edit the result directly; your imported source remains available beside it." : "Run safe cleanup, test a chunk, or process the full source to create a result."}
                    </p>
                  </div>
                  <Button onClick={sendToVoiceStudio} disabled={!hasOutput || isProcessing} className="rounded-xl" data-testid="button-send-to-voice">
                    Continue to audio <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
                <div className="mt-4">
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
              </CardContent>
            </Card>

            {testPreview && (
              <Card className="rounded-2xl border-sky-500/30 bg-sky-500/5 shadow-sm">
                <CardContent className="space-y-4 p-4 sm:p-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline" className="border-sky-500/30 bg-background text-sky-700 dark:text-sky-300">
                          Test preview
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {testPreview.sentenceCount} sentence{testPreview.sentenceCount === 1 ? "" : "s"}
                        </span>
                      </div>
                      <h3 className="mt-2 text-sm font-bold">Compare this sample before processing the full source</h3>
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">
                        This preview is not the final result, is not autosaved as output, and cannot be sent to Voice Studio.
                      </p>
                    </div>
                    <Button type="button" variant="ghost" size="sm" onClick={() => setTestPreview(null)}>
                      Dismiss
                    </Button>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-2">
                    <div className="space-y-2">
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Original sample</p>
                      <Textarea value={testPreview.originalChunk} readOnly className="min-h-[180px] resize-y bg-background" />
                    </div>
                    <div className="space-y-2">
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Processed preview</p>
                      <Textarea value={testPreview.processedChunk} readOnly className="min-h-[180px] resize-y bg-background" />
                    </div>
                  </div>

                  {testPreview.usage && (
                    <p className="text-xs text-muted-foreground">
                      Preview usage: {testPreview.usage.inputTokens.toLocaleString()} input tokens, {testPreview.usage.outputTokens.toLocaleString()} output tokens — ${((testPreview.usage.inputCost || 0) + (testPreview.usage.outputCost || 0)).toFixed(4)}
                    </p>
                  )}
                </CardContent>
              </Card>
            )}

            <div className="min-h-[620px]">
              <OutputDisplay
                text={processedText}
                sourceText={originalText}
                fileName={selectedFile?.name}
                onChange={setProcessedText}
              />
            </div>

            <div className="h-[300px]">
              <ActivityLog logs={logs} onClear={handleClearLogs} />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}


