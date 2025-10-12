import { Card } from "@/components/ui/card";
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
          <Label className="text-sm font-medium">Processing Mode</Label>
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
                <Label
                  htmlFor="mode-none"
                  className="text-sm font-medium cursor-pointer"
                >
                  Single Speaker (No Tags)
                </Label>
                <p className="text-xs text-muted-foreground">
                  Clean text only, without adding any speaker tags
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <RadioGroupItem
                value="format"
                id="mode-format"
                data-testid="radio-mode-format"
              />
              <div className="space-y-0.5">
                <Label
                  htmlFor="mode-format"
                  className="text-sm font-medium cursor-pointer"
                >
                  Format Conversion
                </Label>
                <p className="text-xs text-muted-foreground">
                  Convert existing multi-speaker text to standardized format
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <RadioGroupItem
                value="intelligent"
                id="mode-intelligent"
                data-testid="radio-mode-intelligent"
              />
              <div className="space-y-0.5">
                <Label
                  htmlFor="mode-intelligent"
                  className="text-sm font-medium cursor-pointer"
                >
                  Intelligent Parsing
                </Label>
                <p className="text-xs text-muted-foreground">
                  Detect speakers from prose and structure dialogue output
                </p>
              </div>
            </div>
          </RadioGroup>
        </div>

        {config.mode !== "none" && (
          <>
            <div className="space-y-1.5">
              <Label htmlFor="speaker-count" className="text-sm font-medium">
                Number of Speakers
              </Label>
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
              <Label htmlFor="label-format" className="text-sm font-medium">
                Label Format
              </Label>
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
          </>
        )}
      </div>
    </Card>
  );
}
