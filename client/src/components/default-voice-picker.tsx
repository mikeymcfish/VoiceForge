import { useQuery } from "@tanstack/react-query";
import { Headphones, Library } from "lucide-react";
import type { DefaultVoice, DefaultVoiceCatalog } from "@shared/schema";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const CUSTOM_VALUE = "__custom_voice__";

async function fetchDefaultVoices(): Promise<DefaultVoiceCatalog> {
  const response = await fetch("/api/voices");
  if (!response.ok) throw new Error("Failed to load default voices");
  return response.json();
}

export function DefaultVoicePicker({
  id,
  label = "Default voice library",
  value,
  onChange,
  disabled = false,
  compact = false,
}: {
  id: string;
  label?: string;
  value?: string;
  onChange: (voice: DefaultVoice | undefined) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["default-voices"],
    queryFn: fetchDefaultVoices,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  });
  const selected = data?.voices.find((voice) => voice.id === value);

  return (
    <div className="space-y-2">
      <Label htmlFor={id} className="flex items-center gap-2">
        <Library className="h-4 w-4" />
        {label}
      </Label>
      <Select
        value={value || CUSTOM_VALUE}
        disabled={disabled || isLoading}
        onValueChange={(next) => {
          if (next === CUSTOM_VALUE) onChange(undefined);
          else onChange(data?.voices.find((voice) => voice.id === next));
        }}
      >
        <SelectTrigger id={id}>
          <SelectValue placeholder={isLoading ? "Loading voices…" : "Choose a default voice"} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={CUSTOM_VALUE}>Upload custom audio</SelectItem>
          {(data?.voices ?? []).map((voice) => (
            <SelectItem key={voice.id} value={voice.id}>
              {voice.displayName} · {voice.format.toUpperCase()}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {isError && <p className="text-xs text-destructive">Default voice library is unavailable.</p>}
      {selected && (
        <div className={compact ? "space-y-1" : "rounded-lg border bg-muted/20 p-3"}>
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Headphones className="h-3.5 w-3.5" />
            {selected.hasTranscript ? "Paired transcript will be used for cloning." : "No paired transcript."}
          </p>
          <audio
            controls
            preload="none"
            className="h-8 w-full"
            src={`/api/voices/${encodeURIComponent(selected.id)}/audio`}
          />
        </div>
      )}
    </div>
  );
}

