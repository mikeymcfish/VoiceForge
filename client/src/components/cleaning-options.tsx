import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CircleHelp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { CleaningOptions } from "@shared/schema";

interface CleaningOptionsProps {
  options: CleaningOptions;
  onChange: (options: CleaningOptions) => void;
  disabled?: boolean;
}

const cleaningOptionsList = [
  {
    key: "replaceSmartQuotes" as keyof CleaningOptions,
    label: "Replace smart quotes",
    description: "Convert curly quotes and non-standard punctuation to standard ASCII",
  },
  {
    key: "fixOcrErrors" as keyof CleaningOptions,
    label: "Fix OCR errors",
    description: "Correct spacing issues and merged-word errors from OCR",
  },
  {
    key: "correctSpelling" as keyof CleaningOptions,
    label: "Correct spelling",
    description: "AI-assisted during full processing; not part of the local safe-clean preview",
    aiOnly: true,
  },
  {
    key: "removeUrls" as keyof CleaningOptions,
    label: "Remove URLs",
    description: "Strip web links and URLs from text",
  },
  {
    key: "removeFootnotes" as keyof CleaningOptions,
    label: "Remove footnotes",
    description: "Remove footnote markers and extraneous metadata",
  },
  {
    key: "addPunctuation" as keyof CleaningOptions,
    label: "Add punctuation",
    description: "AI-assisted during full processing; improves headers and loose numbers for TTS prosody",
    aiOnly: true,
  },
  {
    key: "fixHyphenation" as keyof CleaningOptions,
    label: "Fix hyphenation",
    description: "Merge words split by line breaks or hyphens (PDF/EPUB artifacts)",
  },
];

export function CleaningOptionsPanel({
  options,
  onChange,
  disabled,
}: CleaningOptionsProps) {
  const handleChange = (key: keyof CleaningOptions, checked: boolean) => {
    onChange({
      ...options,
      [key]: checked,
    });
  };

  return (
    <Card className="rounded-2xl border-card-border p-4 shadow-sm sm:p-5">
      <h3 className="text-sm font-bold">2. Choose repairs</h3>
      <p className="mb-4 mt-1 text-xs leading-5 text-muted-foreground">Conservative options are enabled by default. Your source is never overwritten.</p>
      <div className="space-y-3">
        {cleaningOptionsList.map((option) => (
          <div key={option.key} className="flex items-start gap-2.5">
            <Checkbox
              id={option.key}
              checked={options[option.key]}
              onCheckedChange={(checked) =>
                handleChange(option.key, checked as boolean)
              }
              disabled={disabled}
              data-testid={`checkbox-${option.key}`}
            />
            <div className="flex-1 space-y-1">
              <div className="flex items-center gap-2">
                <Label
                  htmlFor={option.key}
                  className="text-sm font-medium leading-none cursor-pointer"
                >
                  {option.label}
                </Label>
                {"aiOnly" in option && option.aiOnly && (
                  <Badge variant="outline" className="h-5 rounded-full px-1.5 text-[9px] font-bold uppercase tracking-wide">AI</Badge>
                )}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>{option.description}</TooltipContent>
                </Tooltip>
              </div>
              <p className="text-[11px] leading-4 text-muted-foreground">{option.description}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
