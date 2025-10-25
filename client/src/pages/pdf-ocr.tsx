import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import type {
  PdfOcrJobStatus,
  PdfOcrLogEntry,
  PdfOcrStatus,
  PdfOcrWsMessage,
} from "@shared/schema";
import {
  Check,
  Clipboard,
  Download,
  FileText,
  Loader2,
  Trash2,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";

const statusLabels: Record<PdfOcrJobStatus["status"], string> = {
  queued: "Queued",
  running: "Processing",
  completed: "Completed",
  failed: "Failed",
};

const statusVariants: Record<PdfOcrJobStatus["status"], "default" | "secondary" | "destructive"> = {
  queued: "secondary",
  running: "secondary",
  completed: "default",
  failed: "destructive",
};

type DownloadState = "idle" | "loading";

export default function PdfOcrPage() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [job, setJob] = useState<PdfOcrJobStatus | null>(null);
  const [statusSnapshot, setStatusSnapshot] = useState<PdfOcrStatus | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [logs, setLogs] = useState<PdfOcrLogEntry[]>([]);
  const [combinedText, setCombinedText] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [downloadState, setDownloadState] = useState<DownloadState>("idle");
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");
  const [pageEstimate, setPageEstimate] = useState<number | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const copyTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    activeJobIdRef.current = activeJobId;
  }, [activeJobId]);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/pdf-ocr`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as PdfOcrWsMessage;
        if (message.type === "status") {
          setStatusSnapshot(message.payload);
          const currentId = activeJobIdRef.current;
          if (currentId) {
            const snapshotJob = message.payload.jobs.find((j) => j.id === currentId);
            if (snapshotJob) {
              setJob(snapshotJob);
            }
          }
        } else if (message.type === "job") {
          const nextJob = message.payload;
          setStatusSnapshot((prev) => {
            if (!prev) {
              return prev;
            }
            const jobs = [...prev.jobs];
            const existingIndex = jobs.findIndex((j) => j.id === nextJob.id);
            if (existingIndex >= 0) {
              jobs[existingIndex] = nextJob;
            } else {
              jobs.unshift(nextJob);
            }
            return { ...prev, jobs };
          });
          if (!activeJobIdRef.current || nextJob.id === activeJobIdRef.current) {
            setJob(nextJob);
            if (nextJob.status === "failed") {
              toast({
                title: "PDF OCR failed",
                description: nextJob.error || "See logs for more details.",
                variant: "destructive",
              });
            }
          }
        } else if (message.type === "log") {
          const jobId = message.payload.jobId;
          if (!activeJobIdRef.current || jobId === activeJobIdRef.current) {
            setLogs((prev) => [...prev.slice(-199), message.payload]);
          }
        } else if (message.type === "text") {
          if (!activeJobIdRef.current || message.payload.jobId === activeJobIdRef.current) {
            setCombinedText(message.payload.text);
          }
        }
      } catch (error) {
        console.error("Failed to parse PDF OCR WS message", error);
      }
    };

    ws.onerror = (event) => {
      console.error("PDF OCR WebSocket error", event);
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
      ws.close();
    };
  }, [toast]);

  useEffect(() => {
    if (!job || job.status !== "completed" || combinedText.length > 0) {
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/pdf-ocr/jobs/${job.id}/text`);
        if (!res.ok) {
          return;
        }
        const text = await res.text();
        if (!cancelled) {
          setCombinedText(text);
        }
      } catch (error) {
        console.error("Failed to fetch OCR text", error);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [job, combinedText.length]);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles[0]) {
        setSelectedFile(acceptedFiles[0]);
        setJob(null);
        setActiveJobId(null);
        setLogs([]);
        setCombinedText("");
        setPageEstimate(null);
      }
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    maxFiles: 1,
    disabled: isUploading,
  });

  const isProcessing = job ? job.status === "running" || job.status === "queued" : false;
  const progress = job ? job.progress : 0;
  const processedPages = job?.processedPages ?? 0;
  const totalPages = job?.pageCount ?? pageEstimate ?? undefined;

  const formattedLogs = useMemo(() => {
    return logs
      .slice()
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((entry) => ({
        ...entry,
        date: new Date(entry.timestamp),
      }));
  }, [logs]);

  const handleStart = useCallback(async () => {
    if (!selectedFile) {
      toast({
        title: "No PDF selected",
        description: "Upload a PDF document to begin OCR processing.",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    setLogs([]);
    setCombinedText("");

    try {
      const formData = new FormData();
      formData.append("pdf", selectedFile);

      const response = await fetch("/api/pdf-ocr/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload?.error || "Failed to start PDF OCR");
      }

      const data = await response.json();
      const jobResponse: PdfOcrJobStatus | undefined = data?.job;
      if (jobResponse) {
        setJob(jobResponse);
        setActiveJobId(jobResponse.id);
        activeJobIdRef.current = jobResponse.id;
        setPageEstimate(typeof data?.totalPages === "number" ? data.totalPages : null);
        toast({
          title: "PDF OCR started",
          description: "DeepSeek OCR is processing your document locally.",
        });
      }
    } catch (error) {
      console.error("Failed to start PDF OCR", error);
      toast({
        title: "Failed to start PDF OCR",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  }, [selectedFile, toast]);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setJob(null);
    setActiveJobId(null);
    setLogs([]);
    setCombinedText("");
    setPageEstimate(null);
  }, []);

  const handleCopy = useCallback(async () => {
    if (!combinedText) {
      return;
    }

    try {
      await navigator.clipboard.writeText(combinedText);
      setCopyState("copied");
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
      copyTimeoutRef.current = window.setTimeout(() => {
        setCopyState("idle");
      }, 2000);
      toast({
        title: "Copied",
        description: "OCR output copied to clipboard.",
      });
    } catch (error) {
      console.error("Failed to copy text", error);
      toast({
        title: "Copy failed",
        description: "Unable to copy text to clipboard.",
        variant: "destructive",
      });
    }
  }, [combinedText, toast]);

  const handleDownload = useCallback(() => {
    if (!combinedText || !job) {
      return;
    }

    setDownloadState("loading");
    try {
      const blob = new Blob([combinedText], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const baseName = job.pdfFileName?.replace(/\.pdf$/i, "") || `pdf-ocr-${job.id}`;
      link.href = url;
      link.download = `${baseName}-ocr.txt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast({
        title: "Download started",
        description: "Combined OCR text downloaded.",
      });
    } catch (error) {
      console.error("Failed to download OCR output", error);
      toast({
        title: "Download failed",
        description: "Unable to download OCR output.",
        variant: "destructive",
      });
    } finally {
      setDownloadState("idle");
    }
  }, [combinedText, job, toast]);

  const jobStatusBadge = job ? (
    <Badge variant={statusVariants[job.status]}>{statusLabels[job.status]}</Badge>
  ) : null;

  return (
    <div className="flex h-full flex-col gap-6 overflow-auto p-6">
      <div className="grid gap-6 lg:grid-cols-[400px_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>PDF Document</CardTitle>
            <CardDescription>Upload a PDF and run DeepSeek OCR locally.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              {...getRootProps()}
              className={cn(
                "rounded-lg border-2 border-dashed p-6 transition",
                isDragActive
                  ? "border-primary bg-primary/5"
                  : "border-border bg-muted/40 hover:border-primary/50 hover:bg-muted/60",
                isUploading && "pointer-events-none opacity-70"
              )}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-3 text-center">
                <div className="rounded-full bg-primary/10 p-3 text-primary">
                  <Upload className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-sm font-medium">
                    {isDragActive ? "Drop your PDF here" : "Drag & drop or click to upload a PDF"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">Maximum size 120MB</p>
                </div>
              </div>
            </div>

            {selectedFile && (
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <div className="rounded-md bg-primary/10 p-2 text-primary">
                      <FileText className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm font-medium">{selectedFile.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      {totalPages && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Estimated pages: {totalPages.toLocaleString()}
                        </p>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleClear}
                    disabled={isProcessing || isUploading}
                    aria-label="Clear PDF"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}

            <Button
              onClick={handleStart}
              disabled={!selectedFile || isUploading || isProcessing}
              className="w-full"
            >
              {isUploading || isProcessing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing…
                </>
              ) : (
                "Start PDF OCR"
              )}
            </Button>

            {job && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Status</p>
                    <p className="text-xs text-muted-foreground">
                      {job.message || statusLabels[job.status]}
                    </p>
                  </div>
                  {jobStatusBadge}
                </div>
                <div>
                  <Progress value={progress} aria-label="OCR progress" />
                  <div className="mt-2 text-xs text-muted-foreground">
                    <span>{Math.round(progress)}% complete</span>
                    {typeof totalPages === "number" && (
                      <span className="ml-2">
                        ({processedPages.toLocaleString()} / {totalPages.toLocaleString()} pages)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="flex flex-col">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Progress log</CardTitle>
              <CardDescription>Real-time updates streamed from the local OCR worker.</CardDescription>
            </div>
            {statusSnapshot?.modelsDir && (
              <div className="text-xs text-muted-foreground text-right">
                Models stored at
                <br />
                <code className="font-mono">{statusSnapshot.modelsDir}</code>
              </div>
            )}
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden">
            <div className="h-full rounded-lg border bg-background/40">
              <ScrollArea className="h-full px-4 py-4">
                {formattedLogs.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    {job ? "Waiting for DeepSeek OCR updates…" : "Start a job to view OCR progress."}
                  </p>
                ) : (
                  <ul className="space-y-3 text-sm">
                    {formattedLogs.map((entry) => (
                      <li key={entry.id} className="flex flex-col gap-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium capitalize">{entry.level}</span>
                          <span className="text-xs text-muted-foreground">
                            {entry.date.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-muted-foreground">{entry.message}</p>
                        <Separator />
                      </li>
                    ))}
                  </ul>
                )}
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="flex flex-1 flex-col">
        <CardHeader className="flex flex-row items-center justify-between gap-3">
          <div>
            <CardTitle>Combined OCR text</CardTitle>
            <CardDescription>All pages merged into a single text document.</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleCopy} disabled={!combinedText}>
              {copyState === "copied" ? <Check className="mr-2 h-4 w-4" /> : <Clipboard className="mr-2 h-4 w-4" />}Copy
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              disabled={!combinedText || downloadState === "loading"}
            >
              {downloadState === "loading" ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Download className="mr-2 h-4 w-4" />
              )}
              Download
            </Button>
          </div>
        </CardHeader>
        <CardContent className="flex-1">
          <Textarea
            className="h-full min-h-[300px] resize-none"
            value={combinedText}
            placeholder={
              job
                ? job.status === "completed"
                  ? "OCR finished but no text was produced."
                  : "OCR output will appear here once processing completes."
                : "Start an OCR job to view combined text output."
            }
            readOnly
          />
        </CardContent>
      </Card>
    </div>
  );
}
