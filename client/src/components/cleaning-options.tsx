import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
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
    description: "Fix common spelling mistakes and typos",
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
    description: "Add punctuation after headers and loose numbers for better TTS prosody",
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
    <Card className="p-3">
      <h3 className="text-sm font-medium mb-3">Text Cleaning Options</h3>
      <div className="space-y-2.5">
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
            <div className="flex-1 space-y-0.5">
              <Label
                htmlFor={option.key}
                className="text-sm font-medium leading-none cursor-pointer"
              >
                {option.label}
              </Label>
              <p className="text-xs text-muted-foreground">
                {option.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
