import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import {
  CheckCircle2,
  CircleAlert,
  Cloud,
  CloudDownload,
  Cpu,
  FileText,
  Headphones,
  Laptop,
  PlayCircle,
  RefreshCw,
  Settings2,
  Sparkles,
  Square,
  Waves,
} from "lucide-react";
import type {
  DefaultVoice,
  SpeechEngine,
  SpeechExecutionTarget,
  SpeechJobStatus,
  SpeechOutputFormat,
  SpeechReferenceEnhancement,
  SpeechStatus,
  SpeechWsMessage,
} from "@shared/schema";
import {
  MOSS_DELAY_MODEL_ID,
  MOSS_DURATION_TOKENS_PLACEHOLDER,
  MOSS_LOCAL_CHECKPOINTS,
  mossHostedDurationTokens,
} from "@shared/moss-tts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { HuggingFaceTokenSettings } from "@/components/huggingface-token-settings";
import { DefaultVoicePicker } from "@/components/default-voice-picker";
import { useToast } from "@/hooks/use-toast";

const QWEN_MODELS = [
  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
];
const AUDIO_EXTENSIONS = new Set(["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "webm"]);
const QWEN_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"];
const QWEN_LOCAL_LANGUAGES = [...QWEN_LANGUAGES, "Italian"];
const MOSS_LANGUAGES = [
  "Auto (omit)", "Chinese", "Cantonese", "English", "Arabic", "Czech", "Danish", "Dutch",
  "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Italian", "Japanese",
  "Korean", "Macedonian", "Malay", "Persian (Farsi)", "Polish", "Portuguese", "Romanian", "Russian",
  "Spanish", "Swahili", "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese",
];
const QWEN_SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"];

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default";
  if (status === "failed") return "destructive";
  if (status === "in-progress" || status === "running" || status === "cancelled") return "secondary";
  return "outline";
}

function modeLabel(engine: SpeechEngine, mode: string): string {
  if (engine === "qwen") {
    return { clone: "Voice clone", design: "Voice design", custom: "Preset speaker" }[mode] || mode;
  }
  return {
    direct: "Direct generation",
    clone: "Voice clone",
    continuation: "Continuation",
    "continuation-clone": "Continuation + clone",
  }[mode] || mode;
}

function targetLabel(target: SpeechExecutionTarget): string {
  return target === "local" ? "Local" : "HF ZeroGPU";
}

function referenceEnhancementLabel(mode: SpeechReferenceEnhancement): string {
  return {
    none: "Original reference",
    cleanup: "Gentle FFmpeg cleanup",
    audiosr: "AudioSR super-resolution",
  }[mode];
}

function completedDownloadLabel(job: SpeechJobStatus): string {
  const format = (job.outputFormat ?? "wav").toUpperCase();
  const chapterCount = job.chapterCount ?? 0;
  return chapterCount > 0
    ? `Download ${format} (${chapterCount} ${chapterCount === 1 ? "chapter" : "chapters"})`
    : `Download ${format}`;
}

async function fetchSpeechStatus(): Promise<SpeechStatus> {
  const response = await fetch("/api/speech/status");
  if (!response.ok) throw new Error("Failed to load speech model status");
  return response.json();
}

export function NeuralSpeechPanel({
  engine,
  initialText = "",
}: {
  engine: SpeechEngine;
  initialText?: string;
}) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const config = engine === "qwen"
    ? {
        label: "Qwen3-TTS",
        description: "Fast multilingual voice cloning, voice design, and preset speakers.",
        defaultMode: "clone",
        defaultModel: QWEN_MODELS[0],
        remoteLimit: 1_200,
      }
    : {
        label: "MOSS-TTS v1.5",
        description: "Long-form 31-language speech with cloning, continuation, and pause control.",
        defaultMode: "direct",
        defaultModel: MOSS_DELAY_MODEL_ID,
        remoteLimit: 5_000,
      };

  const { data: status } = useQuery({
    queryKey: ["speech-status"],
    queryFn: fetchSpeechStatus,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  });
  const engineStatus = status?.engines.find((item) => item.engine === engine);
  const jobs = useMemo(
    () => (status?.jobs ?? []).filter((job) => job.engine === engine).sort((a, b) => b.updatedAt - a.updatedAt),
    [engine, status?.jobs]
  );

  const [target, setTarget] = useState<SpeechExecutionTarget>("local");
  const [mode, setMode] = useState(config.defaultMode);
  const [modelId, setModelId] = useState(config.defaultModel);
  const [modelSize, setModelSize] = useState("0.6B");
  const [text, setText] = useState(() => {
    if (initialText.trim()) return initialText;
    try {
      return localStorage.getItem(`vf_${engine}_tts_text_v1`) || "";
    } catch {
      return "";
    }
  });
  const initialTextApplied = useRef(Boolean(text.trim()));
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [defaultVoice, setDefaultVoice] = useState<DefaultVoice | undefined>();
  const voiceInputRef = useRef<HTMLInputElement>(null);
  const [referenceText, setReferenceText] = useState("");
  const [xVectorOnly, setXVectorOnly] = useState(false);
  const [referenceEnhancement, setReferenceEnhancement] =
    useState<SpeechReferenceEnhancement>("none");
  const [audioSrModel, setAudioSrModel] = useState<"speech" | "basic">("speech");
  const [audioSrDevice, setAudioSrDevice] = useState("auto");
  const [audioSrDdimSteps, setAudioSrDdimSteps] = useState(50);
  const [audioSrGuidanceScale, setAudioSrGuidanceScale] = useState(3.5);
  const [audioSrSeed, setAudioSrSeed] = useState(42);
  const [outputFormat, setOutputFormat] = useState<SpeechOutputFormat>("wav");
  const [useChapters, setUseChapters] = useState(false);
  const [chapterPauseMs, setChapterPauseMs] = useState(0);
  const [mp3Quality, setMp3Quality] = useState(2);
  const [language, setLanguage] = useState(engine === "qwen" ? "Auto" : "Auto (omit)");
  const [voiceDescription, setVoiceDescription] = useState("Warm, natural, expressive narration");
  const [speaker, setSpeaker] = useState("Ryan");
  const [instruction, setInstruction] = useState("");
  const [durationControl, setDurationControl] = useState(false);
  const [durationTokens, setDurationTokens] = useState(MOSS_DURATION_TOKENS_PLACEHOLDER);
  const [temperature, setTemperature] = useState(1.7);
  const [topP, setTopP] = useState(0.8);
  const [topK, setTopK] = useState(25);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [maxNewTokens, setMaxNewTokens] = useState(4096);
  const [submitting, setSubmitting] = useState(false);
  const [settingUp, setSettingUp] = useState(false);
  const [connected, setConnected] = useState(false);
  const [cancelling, setCancelling] = useState<Set<string>>(() => new Set());
  const ffmpegAvailable = status?.audioProcessing.ffmpegAvailable === true;
  const audioSrAvailable = status?.audioProcessing.audioSrAvailable === true;
  const effectiveModelId =
    engine === "moss" && target === "hf-space" ? MOSS_DELAY_MODEL_ID : modelId;
  const selectedMossCheckpoint = MOSS_LOCAL_CHECKPOINTS.find(
    (checkpoint) => checkpoint.id === modelId
  );
  const languageOptions =
    engine === "qwen"
      ? target === "local"
        ? QWEN_LOCAL_LANGUAGES
        : QWEN_LANGUAGES
      : MOSS_LANGUAGES;

  useEffect(() => {
    if (!initialTextApplied.current && initialText.trim()) {
      initialTextApplied.current = true;
      setText(initialText);
    }
  }, [initialText]);

  useEffect(() => {
    try {
      if (!text.trim()) localStorage.removeItem(`vf_${engine}_tts_text_v1`);
      else if (text.length <= 750_000) localStorage.setItem(`vf_${engine}_tts_text_v1`, text);
    } catch {}
  }, [engine, text]);

  useEffect(() => {
    const availableModes = target === "local" ? engineStatus?.localModes : engineStatus?.hostedModes;
    if (availableModes?.length && !availableModes.includes(mode)) setMode(availableModes[0]);
  }, [engineStatus?.hostedModes, engineStatus?.localModes, mode, target]);

  useEffect(() => {
    if (!languageOptions.includes(language)) setLanguage(languageOptions[0]);
  }, [language, languageOptions]);

  useEffect(() => {
    if (!status?.audioProcessing) return;
    if (!status.audioProcessing.ffmpegAvailable) {
      if (outputFormat === "mp3") setOutputFormat("wav");
      if (useChapters) setUseChapters(false);
      if (referenceEnhancement === "cleanup") setReferenceEnhancement("none");
    }
    if (!status.audioProcessing.audioSrAvailable && referenceEnhancement === "audiosr") {
      setReferenceEnhancement("none");
    }
  }, [outputFormat, referenceEnhancement, status?.audioProcessing, useChapters]);

  useEffect(() => {
    if (target !== "local" && useChapters) setUseChapters(false);
  }, [target, useChapters]);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${proto}://${window.location.host}/ws/speech`);
    socket.addEventListener("open", () => setConnected(true));
    socket.addEventListener("close", () => setConnected(false));
    socket.addEventListener("error", () => setConnected(false));
    socket.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data) as SpeechWsMessage;
        if (message.type === "status") {
          queryClient.setQueryData(["speech-status"], message.payload);
        } else if (message.type === "job") {
          queryClient.setQueryData<SpeechStatus>(["speech-status"], (previous) => {
            if (!previous) return previous;
            const nextJobs = [...previous.jobs];
            const index = nextJobs.findIndex((job) => job.id === message.payload.id);
            if (index >= 0) nextJobs[index] = message.payload;
            else nextJobs.unshift(message.payload);
            return { ...previous, jobs: nextJobs.slice(0, 40) };
          });
        }
      } catch (error) {
        console.error("Failed to parse speech update", error);
      }
    });
    return () => socket.close();
  }, [queryClient]);

  const localReady = Boolean(
    engineStatus?.runtimeConfigured && engineStatus.availableModels.includes(effectiveModelId)
  );
  const hostedReady = Boolean(status?.tokenConfigured && engineStatus?.hostedAvailable);
  const requiresVoice =
    (engine === "qwen" && mode === "clone") || (engine === "moss" && mode !== "direct");
  const effectiveReferenceEnhancement = requiresVoice ? referenceEnhancement : "none";
  const chapterMarkerCount = (text.match(/\[chapter\]/gi) || []).length;
  const hasChapterContent = text
    .split(/\[chapter\]/i)
    .slice(1)
    .some((section) => section.trim().length > 0);
  const audioSrDeviceValid = /^(?:auto|cpu|mps|cuda(?::\d{1,3})?)$/i.test(
    audioSrDevice.trim()
  );
  const audioProcessingBlocker =
    useChapters && chapterMarkerCount > 500
      ? "A synthesis job can contain at most 500 [CHAPTER] markers"
      : outputFormat === "mp3" && !ffmpegAvailable
      ? "FFmpeg is required for MP3 export"
      : effectiveReferenceEnhancement === "cleanup" && !ffmpegAvailable
        ? "FFmpeg is required for gentle reference cleanup"
        : effectiveReferenceEnhancement === "audiosr" && !audioSrAvailable
          ? "Configure the optional isolated AudioSR tool before using AudioSR enhancement"
          : effectiveReferenceEnhancement === "audiosr" && !audioSrDeviceValid
            ? "Use an AudioSR device such as auto, cpu, mps, cuda, or cuda:0"
          : useChapters && (target !== "local" || !ffmpegAvailable)
            ? "MP3 chapters require Local synthesis with FFmpeg"
          : useChapters && (chapterMarkerCount === 0 || !hasChapterContent)
              ? "Add at least one [CHAPTER] marker followed by spoken text"
            : undefined;
  const characterLimit = target === "hf-space" ? config.remoteLimit : 500_000;
  const ready =
    (target === "local" ? localReady : hostedReady) &&
    text.trim().length > 0 &&
    text.length <= characterLimit &&
    (!requiresVoice || Boolean(voiceFile || defaultVoice)) &&
    !audioProcessingBlocker;
  const readinessMessage = ready
    ? `Ready to run ${targetLabel(target)}`
    : audioProcessingBlocker
      ? audioProcessingBlocker
      : target === "local" && !localReady
      ? "Install the runtime/model and complete required inputs"
      : target === "hf-space" && !hostedReady
        ? "Configure HF access and complete required inputs"
        : "Complete required inputs";

  const handleVoice = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (!file) return setVoiceFile(null);
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !AUDIO_EXTENSIONS.has(extension) || file.size > 32 * 1024 * 1024) {
      event.target.value = "";
      setVoiceFile(null);
      toast({
        title: "Unsupported reference audio",
        description: "Choose a supported audio file no larger than 32 MB.",
        variant: "destructive",
      });
      return;
    }
    setDefaultVoice(undefined);
    setVoiceFile(file);
  };

  const handleDefaultVoice = (voice: DefaultVoice | undefined) => {
    setDefaultVoice(voice);
    if (voice) {
      setVoiceFile(null);
      if (voiceInputRef.current) voiceInputRef.current.value = "";
      if (engine === "qwen" && voice.transcript) setReferenceText(voice.transcript);
    }
  };

  const handleSetup = async () => {
    setSettingUp(true);
    try {
      const response = await fetch(`/api/speech/${engine}/setup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId: effectiveModelId }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload?.error || "Failed to start model download");
      queryClient.setQueryData(["speech-status"], payload);
      toast({ title: "Download started", description: `The pinned ${effectiveModelId} snapshot is being prepared.` });
    } catch (error) {
      toast({
        title: "Setup could not start",
        description: error instanceof Error ? error.message : "Unknown setup error",
        variant: "destructive",
      });
    } finally {
      setSettingUp(false);
    }
  };

  const handleRetrySetup = async () => {
    setSettingUp(true);
    try {
      const stopResponse = await fetch(`/api/speech/${engine}/setup/cancel`, { method: "POST" });
      const stopPayload = await stopResponse.json().catch(() => ({}));
      if (!stopResponse.ok) throw new Error(stopPayload?.error || "Failed to stop the stalled setup");
      const startResponse = await fetch(`/api/speech/${engine}/setup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId: effectiveModelId }),
      });
      const startPayload = await startResponse.json().catch(() => ({}));
      if (!startResponse.ok) throw new Error(startPayload?.error || "Failed to retry model download");
      queryClient.setQueryData(["speech-status"], startPayload);
      toast({ title: "Download restarted", description: "Cached model files were kept and will be reused." });
    } catch (error) {
      toast({
        title: "Retry could not start",
        description: error instanceof Error ? error.message : "Unknown setup error",
        variant: "destructive",
      });
    } finally {
      setSettingUp(false);
    }
  };

  const submit = async (preview: boolean) => {
    if (!ready) return;
    setSubmitting(true);
    try {
      const previewText = text.slice(0, 600).replace(/\s+\S*$/, "").trim();
      const previewSource = previewText || text.trim();
      const previewHasChapterContent = previewSource
        .split(/\[chapter\]/i)
        .slice(1)
        .some((section) => section.trim().length > 0);
      const requestUsesChapters = preview
        ? useChapters && previewHasChapterContent
        : useChapters;
      const submittedText =
        preview && useChapters && !requestUsesChapters
          ? previewSource.replace(/\[chapter\][ \t]*/gi, "")
          : preview
            ? previewSource
            : text;
      const form = new FormData();
      form.append("engine", engine);
      form.append("target", target);
      form.append("mode", mode);
      form.append("text", submittedText);
      form.append("modelId", effectiveModelId);
      form.append("modelSize", modelSize);
      form.append("language", language);
      form.append("referenceText", referenceText);
      form.append("xVectorOnly", String(xVectorOnly));
      form.append("voiceDescription", voiceDescription);
      form.append("speaker", speaker);
      form.append("instruction", instruction);
      form.append("durationControl", String(durationControl));
      form.append(
        "durationTokens",
        String(mossHostedDurationTokens(durationControl, durationTokens))
      );
      form.append("temperature", String(temperature));
      form.append("topP", String(topP));
      form.append("topK", String(topK));
      form.append("repetitionPenalty", String(repetitionPenalty));
      form.append("maxNewTokens", String(maxNewTokens));
      form.append("outputFormat", requestUsesChapters ? "mp3" : outputFormat);
      form.append("useChapters", String(requestUsesChapters));
      form.append("chapterPauseMs", String(chapterPauseMs));
      form.append("mp3Quality", String(mp3Quality));
      form.append("referenceEnhancement", effectiveReferenceEnhancement);
      form.append("audioSrModel", audioSrModel);
      form.append("audioSrDevice", audioSrDevice);
      form.append("audioSrDdimSteps", String(audioSrDdimSteps));
      form.append("audioSrGuidanceScale", String(audioSrGuidanceScale));
      form.append("audioSrSeed", String(audioSrSeed));
      if (voiceFile) form.append("voice", voiceFile);
      else if (defaultVoice) form.append("voiceId", defaultVoice.id);

      const response = await fetch("/api/speech/synthesize", { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload?.error || "Failed to start synthesis");
      toast({
        title: preview ? "Preview queued" : "Synthesis queued",
        description: `${config.label} is running ${target === "local" ? "locally" : "on the official HF Space"}.`,
      });
      if (target === "hf-space") {
        window.setTimeout(() => void queryClient.invalidateQueries({ queryKey: ["huggingface-usage"] }), 2_000);
      }
    } catch (error) {
      toast({
        title: "Synthesis failed",
        description: error instanceof Error ? error.message : "Unable to start synthesis",
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  const cancelJob = async (job: SpeechJobStatus) => {
    setCancelling((previous) => new Set(previous).add(job.id));
    try {
      const response = await fetch(`/api/speech/jobs/${job.id}/cancel`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload?.error || "Cancellation failed");
    } catch (error) {
      toast({
        title: "Cancellation failed",
        description: error instanceof Error ? error.message : "Unable to cancel the job",
        variant: "destructive",
      });
    } finally {
      setCancelling((previous) => {
        const next = new Set(previous);
        next.delete(job.id);
        return next;
      });
    }
  };

  const availableModes = target === "local" ? engineStatus?.localModes ?? [] : engineStatus?.hostedModes ?? [];
  const modelInstalled = engineStatus?.availableModels.includes(effectiveModelId);
  const handleModeChange = (nextMode: string) => {
    setMode(nextMode);
    if (engine === "qwen" && target === "hf-space" && nextMode === "design") {
      setModelSize("1.7B");
      setModelId(QWEN_MODELS[1]);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <Sparkles className="h-5 w-5 text-primary" /> {config.label}
          </h2>
          <p className="text-sm text-muted-foreground">{config.description}</p>
        </div>
        <Badge variant={connected ? "default" : "secondary"}>{connected ? "Realtime connected" : "Polling for updates"}</Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Settings2 className="h-5 w-5 text-primary" />Execution target</CardTitle>
          <CardDescription>
            “Agents” on Hugging Face is the Space API discovery surface. Choose where inference actually runs.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RadioGroup
            value={target}
            onValueChange={(value) => setTarget(value as SpeechExecutionTarget)}
            className="grid gap-3 md:grid-cols-2"
          >
            <Label className={`flex cursor-pointer gap-3 rounded-xl border p-4 ${target === "local" ? "border-primary bg-primary/5" : ""}`}>
              <RadioGroupItem value="local" />
              <Laptop className="h-5 w-5 text-primary" />
              <span><span className="block font-semibold">Local GPU</span><span className="mt-1 block text-xs font-normal text-muted-foreground">Private, long-form, uses the isolated local runtime.</span></span>
            </Label>
            <Label className={`flex cursor-pointer gap-3 rounded-xl border p-4 ${target === "hf-space" ? "border-primary bg-primary/5" : ""}`}>
              <RadioGroupItem value="hf-space" />
              <Cloud className="h-5 w-5 text-primary" />
              <span><span className="block font-semibold">Hugging Face ZeroGPU Space</span><span className="mt-1 block text-xs font-normal text-muted-foreground">Agent/API call to {engineStatus?.spaceId ?? "the official Space"}; consumes your ZeroGPU quota.</span></span>
            </Label>
          </RadioGroup>
        </CardContent>
      </Card>

      {target === "local" ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Cpu className="h-5 w-5 text-primary" />Local runtime</CardTitle>
            <CardDescription>Dependencies stay isolated because Qwen and MOSS require incompatible Transformers versions.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 text-sm sm:grid-cols-3">
              <div><p className="text-muted-foreground">Runtime</p><Badge variant={engineStatus?.runtimeConfigured ? "default" : "destructive"}>{engineStatus?.runtimeConfigured ? "Configured" : "Needs setup"}</Badge></div>
              <div><p className="text-muted-foreground">Selected snapshot</p><Badge variant={modelInstalled ? "default" : "secondary"}>{modelInstalled ? "Installed" : "Not installed"}</Badge></div>
              <div><p className="text-muted-foreground">Download</p><Badge variant={statusVariant(engineStatus?.setupStatus ?? "idle")}>{engineStatus?.setupStatus ?? "idle"}</Badge></div>
            </div>
            {engine === "qwen" && (
              <div className="space-y-2"><Label>Base checkpoint</Label><Select value={modelId} onValueChange={(value) => { setModelId(value); setModelSize(value.includes("0.6B") ? "0.6B" : "1.7B"); }}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{QWEN_MODELS.map((item) => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select></div>
            )}
            {engine === "moss" && (
              <div className="space-y-2">
                <Label>Local checkpoint</Label>
                <Select value={modelId} onValueChange={setModelId}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {MOSS_LOCAL_CHECKPOINTS.map((checkpoint) => (
                      <SelectItem key={checkpoint.id} value={checkpoint.id}>
                        {checkpoint.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedMossCheckpoint && (
                  <p className="text-xs text-muted-foreground">
                    {selectedMossCheckpoint.description}
                  </p>
                )}
              </div>
            )}
            {!engineStatus?.runtimeConfigured && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
                <p className="flex items-center gap-2 font-medium"><CircleAlert className="h-4 w-4 text-amber-600" />Install the isolated runtime first</p>
                <p className="mt-1 text-xs text-muted-foreground">Stop VoiceForge, run <code>VoiceForge.cmd setup-{engine}</code>, restart, then return here to download the pinned model.</p>
              </div>
            )}
            {engineStatus?.lastSetupError && <p className="text-sm text-destructive">{engineStatus.lastSetupError}</p>}
            {engineStatus?.setupStatus === "in-progress" && (
              <div className="space-y-2 rounded-lg border p-3">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span>{engineStatus.setupMessage || "Downloading pinned model..."}</span>
                  <span className="tabular-nums text-muted-foreground">{Math.round(engineStatus.setupProgress ?? 0)}%</span>
                </div>
                <Progress value={engineStatus.setupProgress ?? 0} />
                {engineStatus.setupUpdatedAt && (
                  <p className="text-xs text-muted-foreground">
                    Last activity {formatDistanceToNow(engineStatus.setupUpdatedAt, { addSuffix: true })}. Progress is by setup stage, not downloaded bytes.
                  </p>
                )}
              </div>
            )}
            {engineStatus?.setupStalled && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
                <p className="flex items-center gap-2 font-medium"><CircleAlert className="h-4 w-4 text-amber-600" />Download may be stalled</p>
                <p className="mt-1 text-xs text-muted-foreground">Stop the inactive worker and retry without deleting the cached model files.</p>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleSetup} disabled={settingUp || !engineStatus?.runtimeConfigured || engineStatus?.setupStatus === "in-progress" || modelInstalled}>
                {settingUp || engineStatus?.setupStatus === "in-progress" ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <CloudDownload className="mr-2 h-4 w-4" />}
                Download pinned model
              </Button>
              {engineStatus?.setupStalled && (
                <Button variant="outline" onClick={handleRetrySetup} disabled={settingUp}>
                  {settingUp ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                  Stop & retry
                </Button>
              )}
              <span className="break-all text-xs text-muted-foreground">{engineStatus?.modelsPath}</span>
            </div>
          </CardContent>
        </Card>
      ) : !status?.tokenConfigured ? (
        <Card className="border-amber-500/40">
          <CardHeader><CardTitle>Connect Hugging Face</CardTitle><CardDescription>The token is kept by the local server and is used to attribute ZeroGPU quota to your account.</CardDescription></CardHeader>
          <CardContent><HuggingFaceTokenSettings /></CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><PlayCircle className="h-5 w-5 text-primary" />Generate audio</CardTitle>
          <CardDescription>{target === "local" ? "Run the verified local snapshot." : `Call the official ${config.label} demo through its Gradio API.`}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2"><Label>Mode</Label><Select value={mode} onValueChange={handleModeChange}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{availableModes.map((item) => <SelectItem key={item} value={item}>{modeLabel(engine, item)}</SelectItem>)}</SelectContent></Select></div>
            <div className="space-y-2"><Label>Language</Label><Select value={language} onValueChange={setLanguage}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{languageOptions.map((item) => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select></div>
          </div>

          {engine === "qwen" && target === "hf-space" && mode === "clone" && (
            <div className="max-w-sm space-y-2">
              <Label>Hosted Base model size</Label>
              <Select value={modelSize} onValueChange={(value) => { setModelSize(value); setModelId(value === "0.6B" ? QWEN_MODELS[0] : QWEN_MODELS[1]); }}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent><SelectItem value="0.6B">0.6B (faster)</SelectItem><SelectItem value="1.7B">1.7B (higher capacity)</SelectItem></SelectContent>
              </Select>
            </div>
          )}

          {requiresVoice && (
            <div className="space-y-4">
              <DefaultVoicePicker
                id={`${engine}-default-voice`}
                value={defaultVoice?.id}
                onChange={handleDefaultVoice}
              />
              <div className="space-y-2">
                <Label htmlFor={`${engine}-voice`} className="flex items-center gap-2"><Headphones className="h-4 w-4" />Custom voice reference</Label>
                <Input ref={voiceInputRef} id={`${engine}-voice`} type="file" disabled={Boolean(defaultVoice)} accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.webm" onChange={handleVoice} />
                {voiceFile && <div className="flex items-center justify-between rounded-md border bg-muted/30 px-3 py-2 text-xs"><span className="break-all">{voiceFile.name}</span><Button type="button" variant="ghost" size="sm" onClick={() => { setVoiceFile(null); if (voiceInputRef.current) voiceInputRef.current.value = ""; }}>Clear</Button></div>}
              </div>

              <div className="space-y-3 rounded-xl border p-4">
                <div>
                  <Label>Reference audio enhancement</Label>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Optionally prepare the selected reference before it is sent to {config.label}.
                  </p>
                </div>
                <Select
                  value={referenceEnhancement}
                  onValueChange={(value) =>
                    setReferenceEnhancement(value as SpeechReferenceEnhancement)
                  }
                >
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None - use the original reference</SelectItem>
                    <SelectItem value="cleanup" disabled={!ffmpegAvailable}>
                      Gentle FFmpeg clone prep{ffmpegAvailable ? "" : " (unavailable)"}
                    </SelectItem>
                    <SelectItem value="audiosr" disabled={!audioSrAvailable}>
                      AudioSR super-resolution{audioSrAvailable ? "" : " (unavailable)"}
                    </SelectItem>
                  </SelectContent>
                </Select>

                <div className="flex flex-wrap gap-2">
                  <Badge variant={ffmpegAvailable ? "default" : "secondary"}>
                    FFmpeg {status ? (ffmpegAvailable ? "available" : "unavailable") : "checking"}
                  </Badge>
                  <Badge variant={audioSrAvailable ? "default" : "secondary"}>
                    AudioSR {status ? (audioSrAvailable ? "available" : "unavailable") : "checking"}
                  </Badge>
                </div>

                {referenceEnhancement === "none" && (
                  <p className="text-xs text-muted-foreground">
                    The selected clip is passed through unchanged.
                  </p>
                )}
                {referenceEnhancement === "cleanup" && (
                  <p className="text-xs text-muted-foreground">
                    Gentle clone preparation uses FFmpeg to make a mono 24 kHz WAV with a
                    high-pass filter, light denoise, silence cleanup, loudness normalization,
                    and a final limiter.
                  </p>
                )}
                {referenceEnhancement === "audiosr" && (
                  <div className="space-y-3">
                    <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs text-muted-foreground">
                      AudioSR is optional isolated tooling. It works best on short reference
                      clips; use a clean clip of roughly 5-10 seconds and preview the result
                      before a long render.
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                      <div className="space-y-1">
                        <Label className="text-xs">AudioSR model</Label>
                        <Select
                          value={audioSrModel}
                          onValueChange={(value) => setAudioSrModel(value as "speech" | "basic")}
                        >
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="speech">Speech</SelectItem>
                            <SelectItem value="basic">Basic</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Device</Label>
                        <Input
                          value={audioSrDevice}
                          onChange={(event) => setAudioSrDevice(event.target.value)}
                          placeholder="auto, cpu, cuda:0, or mps"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">DDIM steps</Label>
                        <Input
                          type="number"
                          min={10}
                          max={250}
                          step={1}
                          value={audioSrDdimSteps}
                          onChange={(event) => setAudioSrDdimSteps(Number(event.target.value))}
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Guidance</Label>
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          step={0.1}
                          value={audioSrGuidanceScale}
                          onChange={(event) => setAudioSrGuidanceScale(Number(event.target.value))}
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Seed</Label>
                        <Input
                          type="number"
                          step={1}
                          value={audioSrSeed}
                          onChange={(event) => setAudioSrSeed(Number(event.target.value))}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {!ffmpegAvailable && status && (
                  <p className="text-xs text-muted-foreground">
                    Install FFmpeg and restart VoiceForge to enable gentle cleanup and MP3 export.
                  </p>
                )}
                {!audioSrAvailable && status && (
                  <p className="text-xs text-muted-foreground">
                    AudioSR stays outside the Qwen/MOSS runtimes. Configure
                    <code className="mx-1">VOICEFORGE_AUDIOSR_BIN</code>
                    and restart VoiceForge to enable it.
                  </p>
                )}
              </div>
            </div>
          )}
          {engine === "moss" && (mode === "continuation" || mode === "continuation-clone") && (
            <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs text-muted-foreground">
              Continuation mode expects the reference audio transcript at the beginning of the synthesis text, followed by the new text.
            </p>
          )}

          {engine === "qwen" && mode === "clone" && (
            <div className="space-y-3">
              <div className="space-y-2"><Label>Reference transcript (recommended)</Label><Textarea value={referenceText} onChange={(event) => setReferenceText(event.target.value)} placeholder="Exact words spoken in the reference clip…" className="min-h-20" /></div>
              <Label className="flex items-center gap-2 text-sm"><Checkbox checked={xVectorOnly} onCheckedChange={(value) => setXVectorOnly(value === true)} />Use speaker embedding only (transcript optional, lower fidelity)</Label>
            </div>
          )}
          {engine === "qwen" && mode === "design" && <div className="space-y-2"><Label>Voice description</Label><Textarea value={voiceDescription} onChange={(event) => setVoiceDescription(event.target.value)} placeholder="Describe age, timbre, emotion, pacing, and delivery…" /></div>}
          {engine === "qwen" && mode === "custom" && (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2"><Label>Preset speaker</Label><Select value={speaker} onValueChange={setSpeaker}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{QWEN_SPEAKERS.map((item) => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select></div>
              <div className="space-y-2"><Label>Model size</Label><Select value={modelSize} onValueChange={(value) => { setModelSize(value); setModelId(value === "0.6B" ? QWEN_MODELS[0] : QWEN_MODELS[1]); }}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="0.6B">0.6B</SelectItem><SelectItem value="1.7B">1.7B</SelectItem></SelectContent></Select></div>
              <div className="space-y-2 md:col-span-2"><Label>Style instruction (optional)</Label><Input value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="Cheerful, energetic, calm…" /></div>
            </div>
          )}

          <div className="space-y-2">
            <Label className="flex items-center gap-2"><FileText className="h-4 w-4" />Text to synthesize</Label>
            <Textarea value={text} onChange={(event) => setText(event.target.value)} placeholder={engine === "moss" ? "Paste narration; [pause 1.5s] is supported…" : "Paste the text to speak…"} className="min-h-36" />
            <p className={`text-xs ${text.length > characterLimit ? "text-destructive" : "text-muted-foreground"}`}>{text.length.toLocaleString()} / {characterLimit.toLocaleString()} characters for this target{target === "hf-space" ? " · choose Local for long-form work" : ""}</p>
          </div>

          <div className="space-y-4 rounded-xl border p-4">
            <div>
              <p className="flex items-center gap-2 text-sm font-semibold">
                <Settings2 className="h-4 w-4" />Output audio
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                WAV is lossless. MP3 export and embedded chapter metadata require FFmpeg.
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Output format</Label>
                <Select
                  value={outputFormat}
                  onValueChange={(value) => {
                    const nextFormat = value as SpeechOutputFormat;
                    setOutputFormat(nextFormat);
                    if (nextFormat === "wav") setUseChapters(false);
                  }}
                >
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="wav">WAV (lossless)</SelectItem>
                    <SelectItem value="mp3" disabled={!ffmpegAvailable}>
                      MP3{ffmpegAvailable ? "" : " (FFmpeg unavailable)"}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {outputFormat === "mp3" && !useChapters && (
                <div className="space-y-2">
                  <Label>MP3 quality (0 best, 9 smallest)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={9}
                    step={1}
                    value={mp3Quality}
                    onChange={(event) => setMp3Quality(Number(event.target.value))}
                  />
                </div>
              )}
            </div>

            {target === "local" && ffmpegAvailable ? (
              <div className="space-y-3 rounded-md border bg-muted/20 p-3">
                <Label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={useChapters}
                    onCheckedChange={(value) => {
                      const checked = value === true;
                      setUseChapters(checked);
                      if (checked) setOutputFormat("mp3");
                    }}
                  />
                  Embed MP3 chapters
                </Label>
                <p className="text-xs text-muted-foreground">
                  Put <code>[CHAPTER] Title</code> before each section. The marker starts a
                  chapter at the following rendered audio and <code>Title</code> becomes its
                  MP3 metadata name. {chapterMarkerCount} marker{chapterMarkerCount === 1 ? "" : "s"} detected.
                </p>
                {useChapters && (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Extra chapter pause (ms)</Label>
                      <Input
                        type="number"
                        min={0}
                        max={10_000}
                        step={50}
                        value={chapterPauseMs}
                        onChange={(event) => setChapterPauseMs(Number(event.target.value))}
                      />
                    </div>
                    <p className="self-end text-xs text-muted-foreground">
                      Chaptered MP3 uses chapter-safe encoding for reliable player seeking;
                      the nonchaptered MP3 quality setting is not used.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                Embedded MP3 chapters are available only for Local synthesis with FFmpeg.
              </p>
            )}
          </div>

          {engine === "moss" && (
            <div className="rounded-xl border p-4">
              <p className="mb-3 flex items-center gap-2 text-sm font-semibold"><Settings2 className="h-4 w-4" />MOSS generation controls</p>
              {target === "hf-space" && (
                <div className="mb-4 flex flex-wrap items-center gap-3"><Label className="flex items-center gap-2"><Checkbox checked={durationControl} onCheckedChange={(value) => setDurationControl(value === true)} />Expected-token duration control</Label><Input type="number" className="w-28" min={1} max={4096} value={durationTokens} disabled={!durationControl} onChange={(event) => setDurationTokens(Number(event.target.value))} /></div>
              )}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <div><Label className="text-xs">Temperature</Label><Input type="number" min={0.1} max={3} step={0.05} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></div>
                <div><Label className="text-xs">Top P</Label><Input type="number" min={0.1} max={1} step={0.01} value={topP} onChange={(event) => setTopP(Number(event.target.value))} /></div>
                <div><Label className="text-xs">Top K</Label><Input type="number" min={1} max={200} value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></div>
                <div><Label className="text-xs">Repetition</Label><Input type="number" min={0.5} max={2} step={0.05} value={repetitionPenalty} onChange={(event) => setRepetitionPenalty(Number(event.target.value))} /></div>
                <div><Label className="text-xs">Max tokens</Label><Input type="number" min={128} max={8192} step={128} value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} /></div>
              </div>
            </div>
          )}

          <div className={`rounded-lg border p-3 ${ready ? "border-emerald-500/40 bg-emerald-500/5" : "border-amber-500/40 bg-amber-500/5"}`}>
            <p className="flex items-center gap-2 text-sm font-medium">{ready ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : <CircleAlert className="h-4 w-4 text-amber-600" />}{readinessMessage}</p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <Button variant="outline" onClick={() => void submit(true)} disabled={!ready || submitting}><PlayCircle className="mr-2 h-4 w-4" />Preview first 600 characters</Button>
            <Button onClick={() => void submit(false)} disabled={!ready || submitting}><Waves className="mr-2 h-4 w-4" />Generate audio</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Recent {config.label} jobs</CardTitle><CardDescription>Local and hosted jobs share one queue and output view.</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          {jobs.length === 0 && <p className="text-sm text-muted-foreground">No jobs yet.</p>}
          {jobs.map((job) => (
            <div key={job.id} className="space-y-2 rounded-md border p-3">
              <div className="flex flex-wrap items-start justify-between gap-2"><div><p className="font-medium">Job {job.id}</p><p className="text-xs text-muted-foreground">{targetLabel(job.target)} · {modeLabel(engine, job.mode)} · updated {formatDistanceToNow(job.updatedAt, { addSuffix: true })}</p></div><Badge variant={statusVariant(job.status)}>{job.status}</Badge></div>
              <Progress value={job.progress} />
              <p className="text-xs text-muted-foreground">{job.message || "Waiting…"}{job.queuePosition !== undefined ? ` · queue ${job.queuePosition}` : ""}{job.etaSeconds !== undefined ? ` · ETA ${Math.ceil(job.etaSeconds)}s` : ""}</p>
              {(job.status === "queued" || job.status === "running") && <Button variant="destructive" size="sm" className="w-full" disabled={cancelling.has(job.id)} onClick={() => void cancelJob(job)}>{cancelling.has(job.id) ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <Square className="mr-2 h-4 w-4" />}{cancelling.has(job.id) ? "Cancelling…" : "Cancel job"}</Button>}
              {job.status === "completed" && (
                <div className="space-y-2">
                  <audio controls preload="none" className="w-full" src={`/api/speech/jobs/${job.id}/audio`} />
                  {job.referenceEnhancement && job.referenceEnhancement !== "none" && (
                    <p className="text-xs text-muted-foreground">
                      Reference enhancement: {referenceEnhancementLabel(job.referenceEnhancement)}
                    </p>
                  )}
                  <Button variant="outline" size="sm" className="w-full" asChild>
                    <a href={`/api/speech/jobs/${job.id}/audio?download=1`} download>
                      {completedDownloadLabel(job)}
                    </a>
                  </Button>
                </div>
              )}
              {job.status === "failed" && job.error && <p className="text-xs text-destructive">{job.error}</p>}
            </div>
          ))}
          <Separator />
          <p className="text-xs text-muted-foreground">Hosted demos are mutable community-facing services without an SLA. VoiceForge checks their current API shape before every call and stops safely if it changes.</p>
        </CardContent>
      </Card>
    </div>
  );
}
