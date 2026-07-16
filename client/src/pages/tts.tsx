import { useCallback, useEffect, useMemo, useState } from "react";
import { useRef, type ChangeEvent } from "react";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  DefaultVoice,
  TtsJobStatus,
  TtsStatus,
  TtsWsMessage,
  VibevoiceJobStatus,
  VibevoiceStatus,
  VibevoiceWsMessage,
} from "@shared/schema";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { NeuralSpeechPanel } from "@/components/neural-speech-panel";
import { DefaultVoicePicker } from "@/components/default-voice-picker";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDistanceToNow } from "date-fns";
import {
  BookOpen,
  CheckCircle2,
  CircleAlert,
  CloudDownload,
  Cpu,
  FileText,
  Headphones,
  PlayCircle,
  RefreshCw,
  ShieldCheck,
  Square,
  Sparkles,
  Waves,
} from "lucide-react";

type LogLevel = "info" | "warn" | "error";

interface TtsLogEntry {
  id: string;
  level: LogLevel;
  message: string;
  timestamp: number;
}

const SUPPORTED_AUDIO_EXTENSIONS = new Set([
  "wav",
  "mp3",
  "flac",
  "m4a",
  "aac",
  "ogg",
  "opus",
  "webm",
]);

function hasSupportedAudioExtension(file: File): boolean {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return Boolean(extension && SUPPORTED_AUDIO_EXTENSIONS.has(extension));
}

function indexRuntimeErrorSummary(error: string): string {
  if (error.includes("No module named 'indextts'")) {
    return "The selected Python runtime cannot import the official IndexTTS source.";
  }
  if (error.includes("Microsoft Store")) {
    return "VoiceForge is using the Microsoft Store Python alias instead of an IndexTTS environment.";
  }
  if (error.includes("PyTorch") || error.includes("torch")) {
    return "The selected IndexTTS environment does not meet the required PyTorch runtime checks.";
  }
  return "The IndexTTS runtime check failed. Review the technical details below.";
}

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "completed":
      return "default";
    case "in-progress":
      return "secondary";
    case "failed":
      return "destructive";
    case "cancelled":
      return "secondary";
    default:
      return "outline";
  }
}

function jobStatusLabel(status: TtsJobStatus["status"] | VibevoiceJobStatus["status"]) {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    default:
      return status;
  }
}

function downloadStatusLabel(status: TtsStatus["downloadStatus"]) {
  switch (status) {
    case "idle":
      return "Idle";
    case "in-progress":
      return "Downloading";
    case "completed":
      return "Ready";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

function loadStatusLabel(status: TtsStatus["loadStatus"]) {
  switch (status) {
    case "idle":
      return "Idle";
    case "in-progress":
      return "Checking";
    case "completed":
      return "Ready";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

function setupStatusLabel(status: VibevoiceStatus["setupStatus"]) {
  switch (status) {
    case "idle":
      return "Idle";
    case "in-progress":
      return "Setting up";
    case "completed":
      return "Ready";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

function fetchStatus(): Promise<TtsStatus> {
  return fetch("/api/tts/status").then((res) => {
    if (!res.ok) {
      throw new Error("Failed to fetch TTS status");
    }
    return res.json();
  });
}

function fetchVibeStatus(): Promise<VibevoiceStatus> {
  return fetch("/api/vibevoice/status").then((res) => {
    if (!res.ok) {
      throw new Error("Failed to fetch VibeVoice status");
    }
    return res.json();
  });
}

export default function TtsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<"indextts" | "vibevoice" | "qwen" | "moss">("indextts");

  const { data: status } = useQuery({
    queryKey: ["tts-status"],
    queryFn: fetchStatus,
    refetchInterval: 10000,
    refetchOnWindowFocus: false,
  });

  const { data: vibeStatus } = useQuery({
    queryKey: ["vibevoice-status"],
    queryFn: fetchVibeStatus,
    refetchInterval: 10000,
    refetchOnWindowFocus: false,
  });

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [indexDefaultVoice, setIndexDefaultVoice] = useState<DefaultVoice | undefined>();
  const audioInputRef = useRef<HTMLInputElement>(null);
  const [scriptText, setScriptText] = useState("");
  const [isDownloading, setIsDownloading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [logs, setLogs] = useState<TtsLogEntry[]>([]);
  const [cancellingJobs, setCancellingJobs] = useState<Set<string>>(() => new Set());

  // VibeVoice multi-voice + model state
  const [vibeModelId, setVibeModelId] = useState<string>("");
  const [vibeAudioFile1, setVibeAudioFile1] = useState<File | undefined>(undefined);
  const [vibeAudioFile2, setVibeAudioFile2] = useState<File | undefined>(undefined);
  const [vibeAudioFile3, setVibeAudioFile3] = useState<File | undefined>(undefined);
  const [vibeAudioFile4, setVibeAudioFile4] = useState<File | undefined>(undefined);
  const [vibeDefaultVoices, setVibeDefaultVoices] = useState<Array<DefaultVoice | undefined>>([
    undefined,
    undefined,
    undefined,
    undefined,
  ]);
  const [vibeScriptText, setVibeScriptText] = useState("");
  const [vibeGuidanceScale, setVibeGuidanceScale] = useState(1.3);
  const [isVibeSubmitting, setIsVibeSubmitting] = useState(false);
  const [isVibeSettingUp, setIsVibeSettingUp] = useState(false);
  const [isVibeWsConnected, setIsVibeWsConnected] = useState(false);
  const [vibeLogs, setVibeLogs] = useState<TtsLogEntry[]>([]);
  const [importedDraft, setImportedDraft] = useState(false);

  useEffect(() => {
    const draft = sessionStorage.getItem("vf_tts_draft");
    if (draft?.trim()) {
      setScriptText(draft);
      setVibeScriptText(draft);
      setImportedDraft(true);
      sessionStorage.removeItem("vf_tts_draft");
      return;
    }
    try {
      const saved = JSON.parse(localStorage.getItem("vf_tts_workspace_v1") || "null");
      if (typeof saved?.indexText === "string") setScriptText(saved.indexText);
      if (typeof saved?.vibeText === "string") setVibeScriptText(saved.vibeText);
    } catch {}
  }, []);

  useEffect(() => {
    if (!scriptText.trim() && !vibeScriptText.trim()) {
      try {
        localStorage.removeItem("vf_tts_workspace_v1");
      } catch {}
      return;
    }
    if (scriptText.length > 750_000 || vibeScriptText.length > 750_000) return;
    try {
      localStorage.setItem(
        "vf_tts_workspace_v1",
        JSON.stringify({ indexText: scriptText, vibeText: vibeScriptText, updatedAt: Date.now() })
      );
    } catch {}
  }, [scriptText, vibeScriptText]);

  const sortedJobs = useMemo(() => {
    if (!status?.jobs) return [];
    return [...status.jobs].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [status?.jobs]);

  const vibeSortedJobs = useMemo(() => {
    if (!vibeStatus?.jobs) return [];
    return [...vibeStatus.jobs].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [vibeStatus?.jobs]);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/tts`);

    ws.addEventListener("open", () => setIsWsConnected(true));
    ws.addEventListener("close", () => setIsWsConnected(false));
    ws.addEventListener("error", () => setIsWsConnected(false));

    ws.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(event.data) as TtsWsMessage;
        if (payload.type === "status") {
          queryClient.setQueryData(["tts-status"], payload.payload);
        } else if (payload.type === "job") {
          queryClient.setQueryData(["tts-status"], (previous?: TtsStatus) => {
            if (!previous) {
              return {
                downloadStatus: "idle",
                loadStatus: "idle",
                modelsReady: false,
                runtimeConfigured: false,
                modelsPath: "",
                jobs: [payload.payload],
              } as TtsStatus;
            }
            const nextJobs = [...previous.jobs];
            const index = nextJobs.findIndex((job) => job.id === payload.payload.id);
            if (index >= 0) {
              nextJobs[index] = payload.payload;
            } else {
              nextJobs.unshift(payload.payload);
            }
            return {
              ...previous,
              jobs: nextJobs.slice(0, 20),
            };
          });
        } else if (payload.type === "log") {
          setLogs((prev) => {
            const next = [payload.payload, ...prev];
            if (next.length > 50) {
              next.length = 50;
            }
            return next;
          });
        }
      } catch (error) {
        console.error("Failed to parse TTS WS message", error);
      }
    });

    return () => {
      ws.close();
    };
  }, [queryClient]);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/vibevoice`);

    ws.addEventListener("open", () => setIsVibeWsConnected(true));
    ws.addEventListener("close", () => setIsVibeWsConnected(false));
    ws.addEventListener("error", () => setIsVibeWsConnected(false));

    ws.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(event.data) as VibevoiceWsMessage;
        if (payload.type === "status") {
          queryClient.setQueryData(["vibevoice-status"], payload.payload);
        } else if (payload.type === "job") {
          queryClient.setQueryData(["vibevoice-status"], (previous?: VibevoiceStatus) => {
            if (!previous) {
              return {
                setupStatus: "idle",
                ready: false,
                repoPath: "",
                jobs: [payload.payload],
              } as VibevoiceStatus;
            }
            const nextJobs = [...previous.jobs];
            const index = nextJobs.findIndex((job) => job.id === payload.payload.id);
            if (index >= 0) {
              nextJobs[index] = payload.payload;
            } else {
              nextJobs.unshift(payload.payload);
            }
            return {
              ...previous,
              jobs: nextJobs.slice(0, 20),
            };
          });
        } else if (payload.type === "log") {
          setVibeLogs((prev) => {
            const next = [payload.payload, ...prev];
            if (next.length > 50) {
              next.length = 50;
            }
            return next;
          });
        }
      } catch (error) {
        console.error("Failed to parse VibeVoice WS message", error);
      }
    });

    return () => {
      ws.close();
    };
  }, [queryClient]);

  const handleDownload = useCallback(async () => {
    setIsDownloading(true);
    try {
      const res = await fetch("/api/tts/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        throw new Error(error?.error || "Download request failed");
      }
      toast({
        title: "Download started",
        description: "Model download has been scheduled. Progress will appear below.",
      });
    } catch (error) {
      toast({
        title: "Download failed",
        description: error instanceof Error ? error.message : "Unable to start download",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }, [toast]);

  const handleLoad = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/tts/load", { method: "POST" });
      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        throw new Error(error?.error || "Load request failed");
      }
      toast({
        title: "Runtime check started",
        description: "VoiceForge is verifying the pinned models and isolated Python runtime.",
      });
    } catch (error) {
      toast({
        title: "Runtime check failed",
        description: error instanceof Error ? error.message : "Unable to verify the IndexTTS runtime",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const acceptFileWithinLimit = useCallback((file: File, maxMegabytes: number, label: string) => {
    if (file.size <= maxMegabytes * 1024 * 1024) return true;
    toast({
      title: `${label} is too large`,
      description: `Choose a file no larger than ${maxMegabytes} MB.`,
      variant: "destructive",
    });
    return false;
  }, [toast]);

  const handleAudioChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setAudioFile(null);
      return;
    }
    if (!hasSupportedAudioExtension(file)) {
      setAudioFile(null);
      event.target.value = "";
      toast({
        title: "Unsupported voice reference",
        description: "Choose WAV, MP3, FLAC, M4A, AAC, OGG, Opus, or WebM audio.",
        variant: "destructive",
      });
      return;
    }
    if (!acceptFileWithinLimit(file, 32, "Voice reference")) {
      setAudioFile(null);
      event.target.value = "";
      return;
    }
    setIndexDefaultVoice(undefined);
    setAudioFile(file);
  }, [acceptFileWithinLimit, toast]);

  const clearAudioFile = useCallback(() => {
    setAudioFile(null);
    if (audioInputRef.current) audioInputRef.current.value = "";
  }, []);

  const handleScriptFile = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!acceptFileWithinLimit(file, 2, "Script")) {
        event.target.value = "";
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          if (reader.result.length > 500_000) {
            toast({
              title: "Script is too long",
              description: "Split scripts longer than 500,000 characters into separate render jobs.",
              variant: "destructive",
            });
          } else {
            setScriptText(reader.result);
          }
        }
      };
      reader.readAsText(file);
    }
  }, [acceptFileWithinLimit, toast]);

  const handleSubmit = useCallback(async (preview = false) => {
    if (!audioFile && !indexDefaultVoice) {
      toast({
        title: "Missing audio",
        description: "Select a voice reference audio file before starting synthesis.",
        variant: "destructive",
      });
      return;
    }
    if (!scriptText.trim()) {
      toast({
        title: "Missing text",
        description: "Provide text via file upload or the script editor.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    try {
      const previewText = scriptText
        .slice(0, 600)
        .replace(/\s+\S*$/, "")
        .trim();
      const textToSubmit = preview ? previewText || scriptText.trim() : scriptText;
      const formData = new FormData();
      if (audioFile) formData.append("voice", audioFile);
      else if (indexDefaultVoice) formData.append("voiceId", indexDefaultVoice.id);
      formData.append("text", textToSubmit);

      const res = await fetch("/api/tts/synthesize", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        throw new Error(error?.error || "Failed to start synthesis");
      }

      const data = await res.json();
      const job = data?.job as TtsJobStatus | undefined;

      toast({
        title: preview ? "Preview started" : "Synthesis started",
        description: job
          ? `${preview ? "Short preview" : "Full render"} job ${job.id} is now running.`
          : "Job submitted successfully.",
      });
    } catch (error) {
      toast({
        title: "Synthesis failed",
        description: error instanceof Error ? error.message : "Unable to start synthesis",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [audioFile, indexDefaultVoice, scriptText, toast]);

  const indexRequirements = [
    { label: "Reference voice selected", met: Boolean(audioFile || indexDefaultVoice) },
    { label: "Script contains text", met: scriptText.trim().length > 0 },
    { label: "Pinned model files downloaded", met: Boolean(status?.modelsReady) },
    { label: "IndexTTS runtime configured", met: Boolean(status?.runtimeConfigured) },
    { label: "IndexTTS Python runtime verified", met: status?.loadStatus === "completed" },
  ];
  const missingIndexRequirements = indexRequirements
    .filter((requirement) => !requirement.met)
    .map((requirement) => requirement.label);
  const indexInputsReady = missingIndexRequirements.length === 0;
  const canStart = indexInputsReady && !isSubmitting;
  const canRenderFull = canStart && scriptText.length <= 500_000;
  const indexUnavailableReason = isSubmitting
    ? "A render is already being submitted"
    : `Waiting for: ${missingIndexRequirements.join(", ")}`;

  const handleVibeAudioChange1 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && acceptFileWithinLimit(file, 24, "Voice reference")) {
      setVibeAudioFile1(file);
      setVibeDefaultVoices((voices) => voices.map((voice, index) => index === 0 ? undefined : voice));
    }
    else if (file) event.target.value = "";
  }, [acceptFileWithinLimit]);
  const handleVibeAudioChange2 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && acceptFileWithinLimit(file, 24, "Voice reference")) {
      setVibeAudioFile2(file);
      setVibeDefaultVoices((voices) => voices.map((voice, index) => index === 1 ? undefined : voice));
    }
    else if (file) event.target.value = "";
  }, [acceptFileWithinLimit]);
  const handleVibeAudioChange3 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && acceptFileWithinLimit(file, 24, "Voice reference")) {
      setVibeAudioFile3(file);
      setVibeDefaultVoices((voices) => voices.map((voice, index) => index === 2 ? undefined : voice));
    }
    else if (file) event.target.value = "";
  }, [acceptFileWithinLimit]);
  const handleVibeAudioChange4 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && acceptFileWithinLimit(file, 24, "Voice reference")) {
      setVibeAudioFile4(file);
      setVibeDefaultVoices((voices) => voices.map((voice, index) => index === 3 ? undefined : voice));
    }
    else if (file) event.target.value = "";
  }, [acceptFileWithinLimit]);

  const handleVibeDefaultVoice = useCallback((slot: number, voice: DefaultVoice | undefined) => {
    setVibeDefaultVoices((voices) => voices.map((current, index) => index === slot ? voice : current));
    if (!voice) return;
    if (slot === 0) setVibeAudioFile1(undefined);
    else if (slot === 1) setVibeAudioFile2(undefined);
    else if (slot === 2) setVibeAudioFile3(undefined);
    else if (slot === 3) setVibeAudioFile4(undefined);
  }, []);

  const handleVibeScriptFile = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!acceptFileWithinLimit(file, 2, "Script")) {
        event.target.value = "";
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          if (reader.result.length > 500_000) {
            toast({
              title: "Script is too long",
              description: "Split scripts longer than 500,000 characters into separate render jobs.",
              variant: "destructive",
            });
          } else {
            setVibeScriptText(reader.result);
          }
        }
      };
      reader.readAsText(file);
    }
  }, [acceptFileWithinLimit, toast]);

  const handleVibeSetup = useCallback(async () => {
    setIsVibeSettingUp(true);
    try {
      const res = await fetch("/api/vibevoice/setup", {
        method: "POST",
      });
      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        throw new Error(error?.error || "Setup request failed");
      }
      toast({
        title: "Setup started",
        description: "VibeVoice installation is running. Watch the status card for updates.",
      });
    } catch (error) {
      toast({
        title: "Setup failed",
        description: error instanceof Error ? error.message : "Unable to start VibeVoice setup",
        variant: "destructive",
      });
    } finally {
      setIsVibeSettingUp(false);
    }
  }, [toast]);

  const handleVibeSubmit = useCallback(async (preview = false) => {
    if (!vibeScriptText.trim()) {
      toast({
        title: "Missing text",
        description: "Provide text via file upload or paste it into the editor.",
        variant: "destructive",
      });
      return;
    }

    if (!vibeStatus?.ready) {
      toast({
        title: "VibeVoice not ready",
        description: "Run setup before starting synthesis.",
        variant: "destructive",
      });
      return;
    }

    setIsVibeSubmitting(true);
    try {
      const previewText = vibeScriptText
        .slice(0, 600)
        .replace(/\s+\S*$/, "")
        .trim();
      const textToSubmit = preview ? previewText || vibeScriptText.trim() : vibeScriptText;
      const formData = new FormData();
      if (vibeAudioFile1) formData.append("voice1", vibeAudioFile1);
      else if (vibeDefaultVoices[0]) formData.append("voiceId1", vibeDefaultVoices[0].id);
      if (vibeAudioFile2) formData.append("voice2", vibeAudioFile2);
      else if (vibeDefaultVoices[1]) formData.append("voiceId2", vibeDefaultVoices[1].id);
      if (vibeAudioFile3) formData.append("voice3", vibeAudioFile3);
      else if (vibeDefaultVoices[2]) formData.append("voiceId3", vibeDefaultVoices[2].id);
      if (vibeAudioFile4) formData.append("voice4", vibeAudioFile4);
      else if (vibeDefaultVoices[3]) formData.append("voiceId4", vibeDefaultVoices[3].id);
      formData.append("text", textToSubmit);
      if (Number.isFinite(vibeGuidanceScale)) {
        formData.append("guidanceScale", String(vibeGuidanceScale));
      }
      if (vibeModelId && vibeModelId.trim().length > 0) {
        formData.append("modelId", vibeModelId.trim());
      }

      const res = await fetch("/api/vibevoice/synthesize", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        throw new Error(error?.error || "Failed to start synthesis");
      }

      const data = await res.json();
      const job = data?.job as VibevoiceJobStatus | undefined;

      toast({
        title: preview ? "VibeVoice preview queued" : "VibeVoice job queued",
        description: job
          ? `${preview ? "Short preview" : "Full render"} job ${job.id} is now running.`
          : "Job submitted successfully.",
      });
    } catch (error) {
      toast({
        title: "VibeVoice synthesis failed",
        description: error instanceof Error ? error.message : "Unable to start VibeVoice synthesis",
        variant: "destructive",
      });
    } finally {
      setIsVibeSubmitting(false);
    }
  }, [toast, vibeAudioFile1, vibeAudioFile2, vibeAudioFile3, vibeAudioFile4, vibeDefaultVoices, vibeScriptText, vibeStatus?.ready, vibeGuidanceScale, vibeModelId]);

  const vibeVoiceSlots = [
    vibeAudioFile1 || vibeDefaultVoices[0],
    vibeAudioFile2 || vibeDefaultVoices[1],
    vibeAudioFile3 || vibeDefaultVoices[2],
    vibeAudioFile4 || vibeDefaultVoices[3],
  ];
  const lastVibeVoice = vibeVoiceSlots.reduce((last, file, index) => (file ? index : last), -1);
  const hasContiguousVibeVoices =
    lastVibeVoice >= 0 && !vibeVoiceSlots.slice(0, lastVibeVoice + 1).some((file) => !file);
  const canStartVibe =
    vibeScriptText.trim().length > 0 &&
    !isVibeSubmitting &&
    Boolean(vibeStatus?.ready) &&
    hasContiguousVibeVoices;
  const canRenderFullVibe = canStartVibe && vibeScriptText.length <= 500_000;

  const handleCancelJob = useCallback(async (engine: "tts" | "vibevoice", jobId: string) => {
    const key = `${engine}:${jobId}`;
    setCancellingJobs((previous) => new Set(previous).add(key));
    try {
      const response = await fetch(`/api/${engine}/jobs/${jobId}/cancel`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload?.error || "Failed to cancel synthesis");

      if (engine === "tts") {
        const cancelledJob = payload.job as TtsJobStatus;
        queryClient.setQueryData<TtsStatus>(["tts-status"], (previous) => previous
          ? { ...previous, jobs: previous.jobs.map((job) => job.id === jobId ? cancelledJob : job) }
          : previous);
      } else {
        const cancelledJob = payload.job as VibevoiceJobStatus;
        queryClient.setQueryData<VibevoiceStatus>(["vibevoice-status"], (previous) => previous
          ? { ...previous, jobs: previous.jobs.map((job) => job.id === jobId ? cancelledJob : job) }
          : previous);
      }

      toast({
        title: "Synthesis cancelled",
        description: `Job ${jobId} has been asked to stop.`,
      });
    } catch (error) {
      toast({
        title: "Cancellation failed",
        description: error instanceof Error ? error.message : "Unable to cancel synthesis.",
        variant: "destructive",
      });
    } finally {
      setCancellingJobs((previous) => {
        const next = new Set(previous);
        next.delete(key);
        return next;
      });
    }
  }, [queryClient, toast]);

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6 lg:p-8">
        <header className="flex flex-col gap-4 border-b pb-6 sm:flex-row sm:items-end sm:justify-between">
          <div className="max-w-2xl">
            <Badge variant="secondary" className="mb-3 rounded-full px-3 py-1 text-[11px] font-bold uppercase tracking-[0.15em]">
              <Waves className="mr-1.5 h-3.5 w-3.5 text-primary" /> Voice studio
            </Badge>
            <h1 className="text-3xl font-bold tracking-[-0.04em] sm:text-4xl">Turn the reviewed script into audio.</h1>
            <p className="mt-3 text-sm leading-6 text-muted-foreground sm:text-base">
              Choose an installed engine, add a clean reference voice, and preview a short passage before committing to a long render.
            </p>
            <p className="mt-3 flex items-start gap-2 text-xs leading-5 text-muted-foreground">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
              Use only voice recordings you own or have explicit permission to clone.
            </p>
          </div>
          {importedDraft && (
            <Badge variant="outline" className="w-fit rounded-full border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-emerald-700 dark:text-emerald-300">
              <FileText className="mr-1.5 h-3.5 w-3.5" /> Prepared script imported
            </Badge>
          )}
        </header>
        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as "indextts" | "vibevoice" | "qwen" | "moss")}
          className="space-y-6"
        >
          <TabsList className="h-auto w-full flex-wrap justify-start">
            <TabsTrigger value="indextts" className="flex items-center gap-2">
              <Waves className="h-4 w-4" />
              IndexTTS
            </TabsTrigger>
            <TabsTrigger value="vibevoice" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              VibeVoice
            </TabsTrigger>
            <TabsTrigger value="qwen" className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Qwen3-TTS
            </TabsTrigger>
            <TabsTrigger value="moss" className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              MOSS-TTS v1.5
            </TabsTrigger>
          </TabsList>

          <TabsContent value="indextts" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <Waves className="h-5 w-5 text-primary" />
                  IndexTTS Synthesizer
                </h2>
                <p className="text-sm text-muted-foreground">
                  Download models, prepare prompts, and generate speech with IndexTTS.
                </p>
              </div>
              <Badge variant={isWsConnected ? "default" : "secondary"}>
                {isWsConnected ? "Realtime connected" : "Realtime offline"}
              </Badge>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="h-5 w-5 text-primary" />
                    Model Preparation
                  </CardTitle>
                  <CardDescription>Download the pinned model snapshot and verify the isolated runtime.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Download status</span>
                      <Badge variant={statusVariant(status?.downloadStatus ?? "idle")}>
                        {status ? downloadStatusLabel(status.downloadStatus) : "Unknown"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Runtime check</span>
                      <Badge variant={statusVariant(status?.loadStatus ?? "idle")}>
                        {status ? loadStatusLabel(status.loadStatus) : "Unknown"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Runtime setup</span>
                      <Badge variant={status?.runtimeConfigured ? "default" : "destructive"}>
                        {status?.runtimeConfigured ? "Configured" : "Needs setup"}
                      </Badge>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Models directory</span>
                      <p className="text-xs font-mono mt-1 break-all">{status?.modelsPath ?? "--"}</p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <Button
                      onClick={handleDownload}
                      disabled={isDownloading || status?.downloadStatus === "in-progress" || status?.modelsReady}
                    >
                      <CloudDownload className="h-4 w-4 mr-2" />
                      Download models
                    </Button>
                    <Button
                      onClick={handleLoad}
                      variant="secondary"
                      title={
                        status?.runtimeConfigured
                          ? "Verify the isolated IndexTTS runtime"
                          : "Run VoiceForge.cmd setup-index, then restart VoiceForge"
                      }
                      disabled={
                        isLoading ||
                        !status?.modelsReady ||
                        !status?.runtimeConfigured ||
                        status?.loadStatus === "in-progress"
                      }
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Verify runtime
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Service Overview</CardTitle>
                  <CardDescription>
                    Keep this panel open to monitor status changes while jobs are running.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-muted-foreground">
                  <p>
                    <strong>Models ready:</strong> {status?.modelsReady ? "Yes" : "No"}
                  </p>
                  {status?.lastDownloadError && (
                    <p className="text-destructive">
                      Last download error: {status.lastDownloadError}
                    </p>
                  )}
                  {status && !status.runtimeConfigured && (
                    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-amber-800 dark:text-amber-200">
                      <div className="flex items-start gap-2">
                        <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                        <div className="space-y-1">
                          <p className="font-medium">IndexTTS runtime is not configured</p>
                          <p className="text-xs">
                            Stop VoiceForge, run <code>VoiceForge.cmd setup-index</code>, then restart
                            the app and select Verify runtime. Your existing model download is preserved.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  {status?.lastLoadError && (
                    <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-destructive">
                      <div className="flex items-start gap-2">
                        <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                        <div className="min-w-0 space-y-1">
                          <p className="font-medium">IndexTTS runtime needs setup</p>
                          <p>{indexRuntimeErrorSummary(status.lastLoadError)}</p>
                          <p className="text-xs">
                            Stop VoiceForge and run <code>VoiceForge.cmd setup-index</code> once. The
                            setup creates the supported isolated environment and configures
                            <code className="mx-1">INDEX_TTS_PYTHON</code> and
                            <code>INDEX_TTS_SOURCE_DIR</code>. Restart VoiceForge, then select Verify runtime.
                          </p>
                          <details className="pt-1 text-xs">
                            <summary className="cursor-pointer font-medium">Technical details</summary>
                            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background/70 p-2 font-mono text-foreground">
                              {status.lastLoadError}
                            </pre>
                          </details>
                        </div>
                      </div>
                    </div>
                  )}
                  <Separator />
                  <p>
                    Upload a short reference clip of the target voice and paste the text you want the
                    model to speak. Preview a short passage before committing to the full script.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <PlayCircle className="h-5 w-5 text-primary" />
                    Synthesis Input
                  </CardTitle>
                  <CardDescription>
                    Provide a reference voice sample and the text you want to synthesize.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <DefaultVoicePicker
                    id="index-default-voice"
                    value={indexDefaultVoice?.id}
                    onChange={(voice) => {
                      setIndexDefaultVoice(voice);
                      if (voice) clearAudioFile();
                    }}
                  />
                  <div className="space-y-2">
                    <Label htmlFor="voice-file" className="flex items-center gap-2">
                      <Headphones className="h-4 w-4" />
                      Custom voice reference (audio)
                    </Label>
                    <Input
                      ref={audioInputRef}
                      id="voice-file"
                      type="file"
                      disabled={Boolean(indexDefaultVoice)}
                      accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.webm"
                      onChange={handleAudioChange}
                    />
                    {audioFile && (
                      <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border bg-muted/30 px-3 py-2">
                        <p className="min-w-0 break-all text-xs text-muted-foreground">
                          Selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
                        </p>
                        <Button type="button" variant="ghost" size="sm" onClick={clearAudioFile}>
                          Clear
                        </Button>
                      </div>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="script-file" className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      Script (text file or editor)
                    </Label>
                    <Input id="script-file" type="file" accept=".txt" onChange={handleScriptFile} />
                    <Textarea
                      placeholder="Paste or edit the text to synthesize…"
                      value={scriptText}
                      onChange={(event) => setScriptText(event.target.value)}
                      className="min-h-[120px]"
                    />
                    <p className={`text-xs ${scriptText.length > 500_000 ? "text-destructive" : "text-muted-foreground"}`}>
                      {scriptText.length.toLocaleString()} / 500,000 characters for a full render
                    </p>
                  </div>

                  <div
                    id="index-render-readiness"
                    className={`rounded-lg border p-3 ${
                      indexInputsReady
                        ? "border-emerald-500/40 bg-emerald-500/5"
                        : "border-amber-500/40 bg-amber-500/5"
                    }`}
                  >
                    <p className="mb-2 text-sm font-medium">
                      {indexInputsReady ? "Ready to render" : "Complete these requirements to render"}
                    </p>
                    <ul className="grid gap-1.5 text-sm sm:grid-cols-2">
                      {indexRequirements.map((requirement) => (
                        <li key={requirement.label} className="flex items-center gap-2">
                          {requirement.met ? (
                            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                          ) : (
                            <CircleAlert className="h-4 w-4 shrink-0 text-amber-600" />
                          )}
                          <span className={requirement.met ? "text-foreground" : "text-muted-foreground"}>
                            {requirement.label}
                          </span>
                        </li>
                      ))}
                    </ul>
                    {!indexInputsReady && status?.loadStatus === "failed" && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Runtime verification failed. Stop VoiceForge, run
                        <code className="mx-1">VoiceForge.cmd setup-index</code>, then restart and verify again.
                      </p>
                    )}
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => handleSubmit(true)}
                      disabled={!canStart}
                      aria-describedby="index-render-readiness"
                      title={canStart ? "Render a short preview" : indexUnavailableReason}
                      className="flex items-center justify-center gap-2"
                    >
                      <PlayCircle className="h-4 w-4" />
                      Preview first 600 characters
                    </Button>
                    <Button
                      type="button"
                      onClick={() => handleSubmit(false)}
                      disabled={!canRenderFull}
                      aria-describedby="index-render-readiness"
                      title={
                        canRenderFull
                          ? "Render the full script"
                          : scriptText.length > 500_000
                            ? "Full renders are limited to 500,000 characters"
                            : indexUnavailableReason
                      }
                      className="flex items-center justify-center gap-2"
                    >
                      <Waves className="h-4 w-4" />
                      Render full script
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle>Job Progress</CardTitle>
                  <CardDescription>Track the latest synthesis runs.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {sortedJobs.length === 0 && (
                    <p className="text-sm text-muted-foreground">No synthesis jobs yet.</p>
                  )}
                  {sortedJobs.map((job) => (
                    <div key={job.id} className="rounded-md border p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Job {job.id}</p>
                          <p className="text-xs text-muted-foreground">
                            Updated {formatDistanceToNow(job.updatedAt, { addSuffix: true })}
                          </p>
                        </div>
                        <Badge variant={statusVariant(job.status)}>{jobStatusLabel(job.status)}</Badge>
                      </div>
                      <Progress value={job.progress} />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{job.message || "Waiting…"}</span>
                      </div>
                      {(job.status === "queued" || job.status === "running") && (
                        <Button
                          type="button"
                          variant="destructive"
                          size="sm"
                          className="w-full"
                          disabled={cancellingJobs.has(`tts:${job.id}`)}
                          onClick={() => handleCancelJob("tts", job.id)}
                        >
                          {cancellingJobs.has(`tts:${job.id}`) ? (
                            <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Square className="mr-2 h-4 w-4" />
                          )}
                          {cancellingJobs.has(`tts:${job.id}`) ? "Cancelling…" : "Cancel job"}
                        </Button>
                      )}
                      {job.status === "completed" && job.outputFile && (
                        <div className="space-y-2">
                          <audio
                            controls
                            preload="none"
                            className="w-full"
                            src={`/api/tts/jobs/${job.id}/audio`}
                          />
                          <Button variant="outline" size="sm" className="w-full" asChild>
                            <a href={`/api/tts/jobs/${job.id}/audio?download=1`} download>
                              Download audio
                            </a>
                          </Button>
                        </div>
                      )}
                      {job.status === "failed" && job.error && (
                        <p className="text-xs text-destructive">{job.error}</p>
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle>Event Log</CardTitle>
                  <CardDescription>Latest IndexTTS service events.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {logs.length === 0 && (
                    <p className="text-sm text-muted-foreground">No log entries yet.</p>
                  )}
                  {logs.map((log) => (
                    <div key={log.id} className="text-sm border rounded-md p-3">
                      <div className="flex items-center justify-between mb-1">
                        <Badge
                          variant={
                            log.level === "error"
                              ? "destructive"
                              : log.level === "warn"
                                ? "secondary"
                                : "outline"
                          }
                        >
                          {log.level.toUpperCase()}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(log.timestamp, { addSuffix: true })}
                        </span>
                      </div>
                      <p>{log.message}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="vibevoice" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-primary" />
                  VibeVoice Synthesizer
                </h2>
                <p className="text-sm text-muted-foreground">
                  Prepare the pinned community engine and create long-form, multi-speaker audio.
                </p>
              </div>
              <Badge variant={isVibeWsConnected ? "default" : "secondary"}>
                {isVibeWsConnected ? "Realtime connected" : "Realtime offline"}
              </Badge>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-primary" />
                    Setup & Status
                  </CardTitle>
                  <CardDescription>Install and verify the pinned VibeVoice source locally.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Setup status</span>
                      <Badge variant={statusVariant(vibeStatus?.setupStatus ?? "idle")}>
                        {setupStatusLabel(vibeStatus?.setupStatus ?? "idle")}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Engine ready</span>
                      <Badge variant={vibeStatus?.ready ? "default" : "secondary"}>
                        {vibeStatus?.ready ? "Ready" : "Not ready"}
                      </Badge>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Repository path</span>
                      <p className="text-xs font-mono mt-1 break-all">{vibeStatus?.repoPath ?? "--"}</p>
                    </div>
                    {vibeStatus?.lastSetupError && (
                      <p className="text-xs text-destructive">Last setup error: {vibeStatus.lastSetupError}</p>
                    )}
                  </div>
                  <Separator />
                  <div className="space-y-3">
                    <Button
                      onClick={handleVibeSetup}
                      disabled={isVibeSettingUp || vibeStatus?.setupStatus === "in-progress"}
                      className="w-full"
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Run setup
                    </Button>
                    <p className="text-xs text-muted-foreground">
                      Setup checks out reviewed community revision <code>07cb79feadd2</code>, installs its
                      declared dependencies, and verifies pinned model and tokenizer snapshots. Advanced
                      operators can supply a reviewed, compatible inference command with
                      <code className="mx-1">VIBEVOICE_COMMAND_TEMPLATE</code>.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Usage Tips</CardTitle>
                  <CardDescription>Recommendations for long-form audiobook generation.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-muted-foreground">
                  <p>
                    Voice 1 is required. Additional references map to the first four unique script roles
                    in appearance order; Narrator, Speaker N, and [N] labels are recognized.
                  </p>
                  <p>
                    Guidance scale controls how strongly generation follows the script and voice context.
                    Start at 1.3; larger values may sound less natural.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <PlayCircle className="h-5 w-5 text-primary" />
                    Generate Audio
                  </CardTitle>
                  <CardDescription>
                    Queue a synthesis job using the verified local engine and model snapshot.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="vibe-model">Model</Label>
                    <Select value={vibeModelId} onValueChange={setVibeModelId}>
                      <SelectTrigger id="vibe-model" className="w-full">
                        <SelectValue placeholder="Select model (optional)" />
                      </SelectTrigger>
                      <SelectContent>
                        {(vibeStatus?.availableModels ?? []).map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            {m.id}
                          </SelectItem>
                        ))}
                        {(vibeStatus?.availableModels ?? []).length === 0 && (
                          <SelectItem value="__not-installed" disabled>
                            Run setup to install a model
                          </SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <DefaultVoicePicker id="vibe-default-voice1" label="Voice 1 library (required)" value={vibeDefaultVoices[0]?.id} onChange={(voice) => handleVibeDefaultVoice(0, voice)} compact />
                      <Label htmlFor="vibe-voice1" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 1 custom upload
                      </Label>
                      <Input id="vibe-voice1" type="file" disabled={Boolean(vibeDefaultVoices[0])} accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.webm" onChange={handleVibeAudioChange1} />
                      {vibeAudioFile1 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile1.name}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <DefaultVoicePicker id="vibe-default-voice2" label="Voice 2 library" value={vibeDefaultVoices[1]?.id} onChange={(voice) => handleVibeDefaultVoice(1, voice)} compact />
                      <Label htmlFor="vibe-voice2" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 2 custom upload
                      </Label>
                      <Input id="vibe-voice2" type="file" disabled={Boolean(vibeDefaultVoices[1])} accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.webm" onChange={handleVibeAudioChange2} />
                      {vibeAudioFile2 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile2.name}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <DefaultVoicePicker id="vibe-default-voice3" label="Voice 3 library" value={vibeDefaultVoices[2]?.id} onChange={(voice) => handleVibeDefaultVoice(2, voice)} compact />
                      <Label htmlFor="vibe-voice3" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 3 custom upload
                      </Label>
                      <Input id="vibe-voice3" type="file" disabled={Boolean(vibeDefaultVoices[2])} accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.webm" onChange={handleVibeAudioChange3} />
                      {vibeAudioFile3 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile3.name}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <DefaultVoicePicker id="vibe-default-voice4" label="Voice 4 library" value={vibeDefaultVoices[3]?.id} onChange={(voice) => handleVibeDefaultVoice(3, voice)} compact />
                      <Label htmlFor="vibe-voice4" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 4 custom upload
                      </Label>
                      <Input id="vibe-voice4" type="file" disabled={Boolean(vibeDefaultVoices[3])} accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.webm" onChange={handleVibeAudioChange4} />
                      {vibeAudioFile4 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile4.name}</p>
                      )}
                    </div>
                  </div>
                  <p className="text-xs leading-5 text-muted-foreground">
                    Fill references without gaps. Voice 1 maps to the first unique role in the script,
                    Voice 2 to the second, and so on; the final supplied voice is reused for any missing role.
                  </p>

                  <div className="space-y-2">
                    <Label htmlFor="vibe-script" className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      Script (text file or editor)
                    </Label>
                    <Input id="vibe-script" type="file" accept=".txt" onChange={handleVibeScriptFile} />
                    <Textarea
                      placeholder="Paste the book chapter or narration text…"
                      value={vibeScriptText}
                      onChange={(event) => setVibeScriptText(event.target.value)}
                      className="min-h-[140px]"
                    />
                    <p className={`text-xs ${vibeScriptText.length > 500_000 ? "text-destructive" : "text-muted-foreground"}`}>
                      {vibeScriptText.length.toLocaleString()} / 500,000 characters for a full render
                    </p>
                  </div>

                  <div className="max-w-md space-y-2">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="guidance-slider">Guidance scale</Label>
                        <span className="text-sm font-medium">{vibeGuidanceScale.toFixed(1)}</span>
                      </div>
                      <Slider
                        id="guidance-slider"
                        value={[vibeGuidanceScale]}
                        min={0.5}
                        max={3}
                        step={0.1}
                        onValueChange={(value) => setVibeGuidanceScale(value[0] ?? 1.3)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Classifier-free guidance for the default engine. Recommended: 1.3.
                      </p>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => handleVibeSubmit(true)}
                      disabled={!canStartVibe}
                      className="flex items-center justify-center gap-2"
                    >
                      <PlayCircle className="h-4 w-4" />
                      Preview first 600 characters
                    </Button>
                    <Button
                      type="button"
                      onClick={() => handleVibeSubmit(false)}
                      disabled={!canRenderFullVibe}
                      className="flex items-center justify-center gap-2"
                    >
                      <Waves className="h-4 w-4" />
                      Render full script
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle>Recent Jobs</CardTitle>
                  <CardDescription>Monitor VibeVoice job progress.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {vibeSortedJobs.length === 0 && (
                    <p className="text-sm text-muted-foreground">No VibeVoice jobs yet.</p>
                  )}
                  {vibeSortedJobs.map((job) => (
                    <div key={job.id} className="rounded-md border p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Job {job.id}</p>
                          <p className="text-xs text-muted-foreground">
                            Updated {formatDistanceToNow(job.updatedAt, { addSuffix: true })}
                          </p>
                        </div>
                        <Badge variant={statusVariant(job.status)}>{jobStatusLabel(job.status)}</Badge>
                      </div>
                      <Progress value={job.progress} />
                      <div className="text-xs text-muted-foreground space-y-1">
                        <p>{job.message || "Waiting…"}</p>
                        {job.style && <p>Style: {job.style}</p>}
                        {job.selectedModel && <p>Model: {job.selectedModel}</p>}
                        {job.voiceFileNames && job.voiceFileNames.length > 0 && (
                          <p>Voices: {job.voiceFileNames.join(", ")}</p>
                        )}
                      </div>
                      {(job.status === "queued" || job.status === "running") && (
                        <Button
                          type="button"
                          variant="destructive"
                          size="sm"
                          className="w-full"
                          disabled={cancellingJobs.has(`vibevoice:${job.id}`)}
                          onClick={() => handleCancelJob("vibevoice", job.id)}
                        >
                          {cancellingJobs.has(`vibevoice:${job.id}`) ? (
                            <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Square className="mr-2 h-4 w-4" />
                          )}
                          {cancellingJobs.has(`vibevoice:${job.id}`) ? "Cancelling…" : "Cancel job"}
                        </Button>
                      )}
                      {job.status === "completed" && job.outputFile && (
                        <div className="space-y-2">
                          <audio
                            controls
                            preload="none"
                            className="w-full"
                            src={`/api/vibevoice/jobs/${job.id}/audio`}
                          />
                          <Button variant="outline" size="sm" className="w-full" asChild>
                            <a href={`/api/vibevoice/jobs/${job.id}/audio?download=1`} download>
                              Download audio
                            </a>
                          </Button>
                        </div>
                      )}
                      {job.status === "failed" && job.error && (
                        <p className="text-xs text-destructive">{job.error}</p>
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle>VibeVoice Log</CardTitle>
                  <CardDescription>Real-time worker output and events.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {vibeLogs.length === 0 && (
                    <p className="text-sm text-muted-foreground">No log entries yet.</p>
                  )}
                  {vibeLogs.map((log) => (
                    <div key={log.id} className="text-sm border rounded-md p-3">
                      <div className="flex items-center justify-between mb-1">
                        <Badge
                          variant={
                            log.level === "error"
                              ? "destructive"
                              : log.level === "warn"
                                ? "secondary"
                                : "outline"
                          }
                        >
                          {log.level.toUpperCase()}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(log.timestamp, { addSuffix: true })}
                        </span>
                      </div>
                      <p>{log.message}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="qwen" className="space-y-6">
            <NeuralSpeechPanel engine="qwen" initialText={scriptText || vibeScriptText} />
          </TabsContent>

          <TabsContent value="moss" className="space-y-6">
            <NeuralSpeechPanel engine="moss" initialText={scriptText || vibeScriptText} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

