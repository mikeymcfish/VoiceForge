import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type {
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
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDistanceToNow } from "date-fns";
import {
  BookOpen,
  CloudDownload,
  Cpu,
  FileText,
  Headphones,
  PlayCircle,
  RefreshCw,
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

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "completed":
      return "default";
    case "in-progress":
      return "secondary";
    case "failed":
      return "destructive";
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
      return "Loading";
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
  const [activeTab, setActiveTab] = useState<"indextts" | "vibevoice">("indextts");

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
  const [scriptFile, setScriptFile] = useState<File | null>(null);
  const [scriptText, setScriptText] = useState("");
  const [steps, setSteps] = useState(20);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [logs, setLogs] = useState<TtsLogEntry[]>([]);

  // replaced by multi-voice inputs
  const [vibeScriptFile, setVibeScriptFile] = useState<File | null>(null);
  const [vibeScriptText, setVibeScriptText] = useState("");
  const [vibeStyle, setVibeStyle] = useState("");
  const [vibeTemperature, setVibeTemperature] = useState(0.35);
  const [isVibeSubmitting, setIsVibeSubmitting] = useState(false);
  const [isVibeSettingUp, setIsVibeSettingUp] = useState(false);
  const [isVibeWsConnected, setIsVibeWsConnected] = useState(false);
  const [vibeLogs, setVibeLogs] = useState<TtsLogEntry[]>([]);
  const [vibeRepoUrl, setVibeRepoUrl] = useState("");
  const [vibeRepoBranch, setVibeRepoBranch] = useState("main");

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
        title: "Loading models",
        description: "Model loading has started. Watch the status panel for completion.",
      });
    } catch (error) {
      toast({
        title: "Load failed",
        description: error instanceof Error ? error.message : "Unable to start model load",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const handleAudioChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setAudioFile(file);
    }
  }, []);

  const handleScriptFile = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setScriptFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          setScriptText(reader.result);
        }
      };
      reader.readAsText(file);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!audioFile) {
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
      const formData = new FormData();
      formData.append("voice", audioFile);
      if (scriptFile) {
        formData.append("script", scriptFile);
      } else {
        formData.append("text", scriptText);
      }
      formData.append("steps", String(steps));

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
        title: "Synthesis started",
        description: job ? `Job ${job.id} is now running.` : "Job submitted successfully.",
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
  }, [audioFile, scriptFile, scriptText, steps, toast]);

  const canStart =
    Boolean(audioFile) && scriptText.trim().length > 0 && !isSubmitting && status?.downloadStatus === "completed";

  const handleVibeAudioChange1 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setVibeAudioFile1(file);
  }, []);
  const handleVibeAudioChange2 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setVibeAudioFile2(file);
  }, []);
  const handleVibeAudioChange3 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setVibeAudioFile3(file);
  }, []);
  const handleVibeAudioChange4 = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setVibeAudioFile4(file);
  }, []);

  const handleVibeScriptFile = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVibeScriptFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          setVibeScriptText(reader.result);
        }
      };
      reader.readAsText(file);
    }
  }, []);

  const handleVibeSetup = useCallback(async () => {
    setIsVibeSettingUp(true);
    try {
      const payload: Record<string, string> = {};
      if (vibeRepoUrl.trim().length > 0) {
        payload.repoUrl = vibeRepoUrl.trim();
      }
      if (vibeRepoBranch.trim().length > 0) {
        payload.branch = vibeRepoBranch.trim();
      }
      const res = await fetch("/api/vibevoice/setup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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
  }, [toast, vibeRepoBranch, vibeRepoUrl]);

  const handleVibeSubmit = useCallback(async () => {
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
      const formData = new FormData();
      if (vibeAudioFile1) formData.append("voice1", vibeAudioFile1);
      if (vibeAudioFile2) formData.append("voice2", vibeAudioFile2);
      if (vibeAudioFile3) formData.append("voice3", vibeAudioFile3);
      if (vibeAudioFile4) formData.append("voice4", vibeAudioFile4);
      if (vibeScriptFile) {
        formData.append("script", vibeScriptFile);
      } else {
        formData.append("text", vibeScriptText);
      }
      if (vibeStyle.trim().length > 0) {
        formData.append("style", vibeStyle.trim());
      }
      if (Number.isFinite(vibeTemperature)) {
        formData.append("temperature", String(vibeTemperature));
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
        title: "VibeVoice job queued",
        description: job ? `Job ${job.id} is now running.` : "Job submitted successfully.",
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
  }, [toast, vibeAudioFile1, vibeAudioFile2, vibeAudioFile3, vibeAudioFile4, vibeScriptFile, vibeScriptText, vibeStatus?.ready, vibeStyle, vibeTemperature, vibeModelId]);

  const canStartVibe = vibeScriptText.trim().length > 0 && !isVibeSubmitting && Boolean(vibeStatus?.ready);

  // VibeVoice state: model + up to 4 voices
  const [vibeModelId, setVibeModelId] = useState<string>("");
  const [vibeAudioFile1, setVibeAudioFile1] = useState<File | undefined>(undefined);
  const [vibeAudioFile2, setVibeAudioFile2] = useState<File | undefined>(undefined);
  const [vibeAudioFile3, setVibeAudioFile3] = useState<File | undefined>(undefined);
  const [vibeAudioFile4, setVibeAudioFile4] = useState<File | undefined>(undefined);

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-6xl mx-auto p-6 space-y-6">
        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as "indextts" | "vibevoice")}
          className="space-y-6"
        >
          <TabsList className="w-full justify-start">
            <TabsTrigger value="indextts" className="flex items-center gap-2">
              <Waves className="h-4 w-4" />
              IndexTTS
            </TabsTrigger>
            <TabsTrigger value="vibevoice" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              VibeVoice
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
                  <CardDescription>Download and load the IndexTTS repository models.</CardDescription>
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
                      <span className="text-muted-foreground">Model load</span>
                      <Badge variant={statusVariant(status?.loadStatus ?? "idle")}>
                        {status ? loadStatusLabel(status.loadStatus) : "Unknown"}
                      </Badge>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Models directory</span>
                      <p className="text-xs font-mono mt-1 break-all">{status?.modelsPath ?? "--"}</p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <Button onClick={handleDownload} disabled={isDownloading || status?.downloadStatus === "in-progress"}>
                      <CloudDownload className="h-4 w-4 mr-2" />
                      Download models
                    </Button>
                    <Button
                      onClick={handleLoad}
                      variant="secondary"
                      disabled={
                        isLoading ||
                        status?.downloadStatus !== "completed" ||
                        status?.loadStatus === "in-progress"
                      }
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Load models
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
                  {status?.lastLoadError && (
                    <p className="text-destructive">
                      Last load error: {status.lastLoadError}
                    </p>
                  )}
                  <Separator />
                  <p>
                    Upload a short reference clip of the target voice and paste the text you want the
                    model to speak. Increase the diffusion steps for higher fidelity at the cost of
                    runtime.
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
                  <div className="space-y-2">
                    <Label htmlFor="voice-file" className="flex items-center gap-2">
                      <Headphones className="h-4 w-4" />
                      Voice reference (audio)
                    </Label>
                    <Input id="voice-file" type="file" accept="audio/*" onChange={handleAudioChange} />
                    {audioFile && (
                      <p className="text-xs text-muted-foreground">
                        Selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
                      </p>
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
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="steps-slider">Generation steps</Label>
                      <span className="text-sm font-medium">{steps} steps</span>
                    </div>
                    <Slider
                      id="steps-slider"
                      value={[steps]}
                      min={20}
                      max={50}
                      step={1}
                      onValueChange={(value) => setSteps(value[0] ?? 20)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Controls the diffusion refinement steps (higher values improve quality but take
                      longer).
                    </p>
                  </div>

                  <Button
                    onClick={handleSubmit}
                    disabled={!canStart}
                    className="w-full flex items-center justify-center gap-2"
                  >
                    <PlayCircle className="h-4 w-4" />
                    Start synthesis
                  </Button>
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
                        {job.steps && <span>{job.steps} steps</span>}
                      </div>
                      {job.status === "completed" && job.outputFile && (
                        <Button variant="outline" size="sm" className="w-full" asChild>
                          <a href={`/api/tts/jobs/${job.id}/audio`} target="_blank" rel="noreferrer">
                            Download audio
                          </a>
                        </Button>
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
                  Clone the VibeVoice community project and trigger audiobook-style synthesis runs.
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
                  <CardDescription>
                    Configure the Git repository to pull and install VibeVoice locally.
                  </CardDescription>
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
                    <div className="space-y-1">
                      <Label htmlFor="vibe-repo">Repository URL (optional)</Label>
                      <Input
                        id="vibe-repo"
                        placeholder="https://github.com/vibevoice-community/VibeVoice.git"
                        value={vibeRepoUrl}
                        onChange={(event) => setVibeRepoUrl(event.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="vibe-branch">Branch</Label>
                      <Input
                        id="vibe-branch"
                        value={vibeRepoBranch}
                        onChange={(event) => setVibeRepoBranch(event.target.value)}
                      />
                    </div>
                    <Button
                      onClick={handleVibeSetup}
                      disabled={isVibeSettingUp || vibeStatus?.setupStatus === "in-progress"}
                      className="w-full"
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Run setup
                    </Button>
                    <p className="text-xs text-muted-foreground">
                      The worker clones the repository, installs dependencies, and runs any bundled
                      asset download script. Override the command with the
                      <code className="mx-1">VIBEVOICE_COMMAND_TEMPLATE</code> environment variable if
                      the default inference script differs in your fork.
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
                    Provide a clean reference clip for the narrator voice. If you skip the voice file,
                    the configured template will determine how VibeVoice selects speakers.
                  </p>
                  <p>
                    Styles and temperature controls are forwarded directly to the worker command. Adjust
                    them to tune expressiveness while keeping pronunciation stable.
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
                    Queue a VibeVoice synthesis job using the configured repository and command template.
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
                        {(
                          (vibeStatus?.availableModels ?? []).length > 0
                            ? vibeStatus?.availableModels
                            : [
                                { id: "microsoft/VibeVoice-1.5B", path: "" },
                                { id: "aoi-ot/VibeVoice-Large", path: "" },
                              ]
                        ).map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            {m.id}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="vibe-voice1" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 1 (optional)
                      </Label>
                      <Input id="vibe-voice1" type="file" accept="audio/*" onChange={handleVibeAudioChange1} />
                      {vibeAudioFile1 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile1.name}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="vibe-voice2" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 2 (optional)
                      </Label>
                      <Input id="vibe-voice2" type="file" accept="audio/*" onChange={handleVibeAudioChange2} />
                      {vibeAudioFile2 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile2.name}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="vibe-voice3" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 3 (optional)
                      </Label>
                      <Input id="vibe-voice3" type="file" accept="audio/*" onChange={handleVibeAudioChange3} />
                      {vibeAudioFile3 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile3.name}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="vibe-voice4" className="flex items-center gap-2">
                        <Headphones className="h-4 w-4" />
                        Voice 4 (optional)
                      </Label>
                      <Input id="vibe-voice4" type="file" accept="audio/*" onChange={handleVibeAudioChange4} />
                      {vibeAudioFile4 && (
                        <p className="text-xs text-muted-foreground">Selected: {vibeAudioFile4.name}</p>
                      )}
                    </div>
                  </div>

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
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="vibe-style">Style tag (optional)</Label>
                      <Input
                        id="vibe-style"
                        placeholder="e.g. emotional, storyteller"
                        value={vibeStyle}
                        onChange={(event) => setVibeStyle(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="temperature-slider">Temperature</Label>
                        <span className="text-sm font-medium">{vibeTemperature.toFixed(2)}</span>
                      </div>
                      <Slider
                        id="temperature-slider"
                        value={[vibeTemperature]}
                        min={0}
                        max={1}
                        step={0.05}
                        onValueChange={(value) => setVibeTemperature(value[0] ?? 0.35)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Lower values keep speech stable. Increase slightly for more expressive delivery.
                      </p>
                    </div>
                  </div>

                  <Button
                    onClick={handleVibeSubmit}
                    disabled={!canStartVibe}
                    className="w-full flex items-center justify-center gap-2"
                  >
                    <PlayCircle className="h-4 w-4" />
                    Start VibeVoice synthesis
                  </Button>
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
                      {job.status === "completed" && job.outputFile && (
                        <Button variant="outline" size="sm" className="w-full" asChild>
                          <a href={`/api/vibevoice/jobs/${job.id}/audio`} target="_blank" rel="noreferrer">
                            Download audio
                          </a>
                        </Button>
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
        </Tabs>
      </div>
    </div>
  );
}
