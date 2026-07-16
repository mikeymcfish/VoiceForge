import { Card } from "@/components/ui/card";
import { CircleHelp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { SpeakerConfig } from "@shared/schema";

interface SpeakerConfigProps {
  config: SpeakerConfig;
  onChange: (config: SpeakerConfig) => void;
  disabled?: boolean;
}

export function SpeakerConfigPanel({
  config,
  onChange,
  disabled,
}: SpeakerConfigProps) {
  const isSingleSpeakerMode = config.mode === "none";
  const isIntelligentMode = config.mode === "intelligent";
  const characterMapping = config.characterMapping ?? [];

  return (
    <Card className="rounded-2xl border-card-border p-4 shadow-sm sm:p-5">
      <h3 className="text-sm font-bold">3. Choose voice structure</h3>
      <p className="mb-4 mt-1 text-xs leading-5 text-muted-foreground">Keep one voice, normalize existing labels, or ask AI to cast dialogue.</p>

      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label className="text-sm font-medium">Desired outcome</Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>Select how to handle speaker tagging</TooltipContent>
            </Tooltip>
          </div>
          <RadioGroup
            value={config.mode}
            onValueChange={(value) =>
              onChange({ ...config, mode: value as "none" | "format" | "intelligent" })
            }
            disabled={disabled}
            className="space-y-2.5"
          >
            <div className="flex items-start gap-2.5">
              <RadioGroupItem
                value="none"
                id="mode-none"
                data-testid="radio-mode-none"
              />
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="mode-none" className="text-sm font-medium cursor-pointer">
                    Single voice
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Clean text only; do not add any speaker labels</TooltipContent>
                  </Tooltip>
                </div>
                <p className="text-[11px] leading-4 text-muted-foreground">Keep the prose intact without adding speaker tags.</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <RadioGroupItem
                value="format"
                id="mode-format"
                data-testid="radio-mode-format"
              />
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="mode-format" className="text-sm font-medium cursor-pointer">
                    Preserve existing speakers
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Convert existing multi-speaker text into your selected label style</TooltipContent>
                  </Tooltip>
                </div>
                <p className="text-[11px] leading-4 text-muted-foreground">Normalize a script that already identifies each speaker.</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <RadioGroupItem
                value="intelligent"
                id="mode-intelligent"
                data-testid="radio-mode-intelligent"
              />
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="mode-intelligent" className="text-sm font-medium cursor-pointer">
                    Detect dialogue & speakers
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Use the LLM to detect speakers and structure dialogue</TooltipContent>
                  </Tooltip>
                </div>
                <p className="text-[11px] leading-4 text-muted-foreground">Identify quoted speech, narration, and stable character labels.</p>
              </div>
            </div>
          </RadioGroup>
        </div>

        {!isSingleSpeakerMode && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Label htmlFor="speaker-count" className="text-sm font-medium">
                  Available voice slots
                </Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>Target number of speakers for formatting/detection</TooltipContent>
                </Tooltip>
              </div>
              <Input
                id="speaker-count"
                type="number"
                min={1}
                max={20}
                value={config.speakerCount}
                onChange={(e) =>
                  onChange({
                    ...config,
                    speakerCount: parseInt(e.target.value) || 2,
                  })
                }
                disabled={disabled}
                data-testid="input-speaker-count"
                className="h-9"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Label htmlFor="label-format" className="text-sm font-medium">
                  Label Format
                </Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>Choose "Speaker N:" or numeric labels like "[N]:"</TooltipContent>
                </Tooltip>
              </div>
              <Select
                value={config.labelFormat}
                onValueChange={(value) =>
                  onChange({
                    ...config,
                    labelFormat: value as "speaker" | "bracket",
                  })
                }
                disabled={disabled}
              >
                <SelectTrigger
                  id="label-format"
                  data-testid="select-label-format"
                  className="h-10"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="speaker">Speaker 1:, Speaker 2:, ...</SelectItem>
                  <SelectItem value="bracket">[1]:, [2]:, [3]:, ...</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {isIntelligentMode && !config.includeNarrator && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-[11px] leading-5 text-amber-800 dark:text-amber-200">
                Dialogue-only mode removes descriptive prose from the result. Enable narration in character detection to preserve it.
              </div>
            )}

            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Label htmlFor="narrator-attr" className="text-sm font-medium">
                  Narrator Attribution
                </Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>
                    Applies when using Intelligent Parsing and including a Narrator.
                  </TooltipContent>
                </Tooltip>
              </div>
              <Select
                value={config.narratorAttribution ?? "remove"}
                onValueChange={(value) =>
                  onChange({
                    ...config,
                    narratorAttribution: value as "remove" | "verbatim" | "contextual",
                  })
                }
                disabled={disabled || !isIntelligentMode}
              >
                <SelectTrigger id="narrator-attr" className="h-10">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="remove">Remove attribution tags</SelectItem>
                  <SelectItem value="verbatim">Narrator says tags (verbatim)</SelectItem>
                  <SelectItem value="contextual">Narrator adds context (intelligent)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {config.includeNarrator && characterMapping.length > 0 && (
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Select Narrator (Speaker 1)</Label>
                <RadioGroup
                  value={config.narratorCharacterName ?? ""}
                  onValueChange={(val) =>
                    onChange({
                      ...config,
                      narratorCharacterName: val || undefined,
                    })
                  }
                >
                  <div className="flex items-center gap-2 py-1">
                    <RadioGroupItem value="" id="narrator-none" />
                    <Label htmlFor="narrator-none" className="text-sm cursor-pointer">None</Label>
                  </div>
                  {characterMapping.map((c) => (
                    <div key={c.speakerNumber} className="flex items-center gap-2 py-1">
                      <RadioGroupItem value={c.name} id={`narrator-${c.speakerNumber}`} />
                      <Label htmlFor={`narrator-${c.speakerNumber}`} className="text-sm cursor-pointer">
                        Speaker {c.speakerNumber}: {c.name}
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
}
