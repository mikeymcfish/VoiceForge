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
  return (
    <Card className="p-3">
      <h3 className="text-sm font-medium mb-3">Multi-Speaker Configuration</h3>
      
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label className="text-sm font-medium">Processing Mode</Label>
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
                    Single Speaker (No Tags)
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Clean text only; do not add any speaker labels</TooltipContent>
                  </Tooltip>
                </div>
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
                    Format Conversion
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Convert existing multi-speaker text into your selected label style</TooltipContent>
                  </Tooltip>
                </div>
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
                    Intelligent Parsing
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Use the LLM to detect speakers and structure dialogue</TooltipContent>
                  </Tooltip>
                </div>
              </div>
            </div>
          </RadioGroup>
        </div>

        {config.mode !== "none" && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Label htmlFor="speaker-count" className="text-sm font-medium">
                  Number of Speakers
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
                value={(config as any).narratorAttribution || 'remove'}
                onValueChange={(value) =>
                  onChange({
                    ...config,
                    // @ts-expect-error - extend SpeakerConfig shape
                    narratorAttribution: value as 'remove' | 'verbatim' | 'contextual',
                  })
                }
                disabled={disabled || config.mode === 'none'}
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

            {config.includeNarrator && (config.characterMapping?.length || 0) > 0 && (
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Select Narrator (Speaker 1)</Label>
                <RadioGroup
                  value={(config as any).narratorCharacterName || ""}
                  onValueChange={(val) => onChange({ ...(config as any), narratorCharacterName: val || undefined } as any)}
                >
                  <div className="flex items-center gap-2 py-1">
                    <RadioGroupItem value="" id="narrator-none" />
                    <Label htmlFor="narrator-none" className="text-sm cursor-pointer">None</Label>
                  </div>
                  {config.characterMapping!.map((c) => (
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
