import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { TtsJobStatus, TtsStatus, TtsWsMessage } from "@shared/schema";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { formatDistanceToNow } from "date-fns";
import { CloudDownload, Cpu, FileText, Headphones, PlayCircle, RefreshCw, Waves } from "lucide-react";

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

function jobStatusLabel(status: TtsJobStatus["status"]) {
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

function fetchStatus(): Promise<TtsStatus> {
  return fetch("/api/tts/status").then((res) => {
    if (!res.ok) {
      throw new Error("Failed to fetch TTS status");
    }
    return res.json();
  });
}

export default function TtsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: status } = useQuery({
    queryKey: ["tts-status"],
    queryFn: fetchStatus,
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

  const sortedJobs = useMemo(() => {
    if (!status?.jobs) return [];
    return [...status.jobs].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [status?.jobs]);

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
        description: job
          ? `Job ${job.id} is now running.`
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
  }, [audioFile, scriptFile, scriptText, steps, toast]);

  const canStart =
    Boolean(audioFile) && scriptText.trim().length > 0 && !isSubmitting && status?.downloadStatus === "completed";

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-6xl mx-auto p-6 space-y-6">
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
                <CloudDownload className="h-5 w-5 text-primary" />
                Model Setup
              </CardTitle>
              <CardDescription>Prepare IndexTTS assets before running synthesis.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="font-medium">Model download</p>
                  <p className="text-sm text-muted-foreground">
                    Status: {downloadStatusLabel(status?.downloadStatus || "idle")}
                  </p>
                </div>
                <Badge variant={statusVariant(status?.downloadStatus || "idle")}>
                  {downloadStatusLabel(status?.downloadStatus || "idle")}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="font-medium">Model load</p>
                  <p className="text-sm text-muted-foreground">
                    Status: {loadStatusLabel(status?.loadStatus || "idle")}
                  </p>
                </div>
                <Badge variant={statusVariant(status?.loadStatus || "idle")}>
                  {loadStatusLabel(status?.loadStatus || "idle")}
                </Badge>
              </div>
              {status?.lastDownloadError && (
                <p className="text-sm text-destructive">{status.lastDownloadError}</p>
              )}
              {status?.lastLoadError && (
                <p className="text-sm text-destructive">{status.lastLoadError}</p>
              )}
              <Separator />
              <div className="flex flex-wrap gap-3">
                <Button
                  variant="default"
                  onClick={handleDownload}
                  disabled={isDownloading || status?.downloadStatus === "in-progress"}
                  className="flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  {status?.downloadStatus === "completed" ? "Re-download" : "Download models"}
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleLoad}
                  disabled={
                    isLoading ||
                    status?.downloadStatus !== "completed" ||
                    status?.loadStatus === "in-progress"
                  }
                  className="flex items-center gap-2"
                >
                  <Cpu className="h-4 w-4" />
                  Load models
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Models directory: <span className="font-mono">{status?.modelsPath || "pending"}</span>
              </p>
            </CardContent>
          </Card>

          <Card>
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
                <Input
                  id="voice-file"
                  type="file"
                  accept="audio/*"
                  onChange={handleAudioChange}
                />
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
                <Input
                  id="script-file"
                  type="file"
                  accept=".txt"
                  onChange={handleScriptFile}
                />
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
                  Controls the diffusion refinement steps (higher values improve quality but take longer).
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
                    <Badge variant={statusVariant(job.status)}>
                      {jobStatusLabel(job.status)}
                    </Badge>
                  </div>
                  <Progress value={job.progress} />
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{job.message || "Waiting…"}</span>
                    {job.steps && <span>{job.steps} steps</span>}
                  </div>
                  {job.status === "completed" && job.outputFile && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full"
                      asChild
                    >
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
                    <Badge variant={log.level === "error" ? "destructive" : log.level === "warn" ? "secondary" : "outline"}>
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
      </div>
    </div>
  );
}
