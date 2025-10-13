import { Copy, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useMemo } from "react";
import { useToast } from "@/hooks/use-toast";

interface OutputDisplayProps {
  text: string;
  fileName?: string;
}

export function OutputDisplay({ text, fileName }: OutputDisplayProps) {
  const { toast } = useToast();

  const coloredLines = useMemo(() => {
    const lines = (text || "").split(/\r?\n/);
    // Bright colors that pop on black background
    const palette = [
      "text-blue-400",
      "text-emerald-400",
      "text-violet-400",
      "text-amber-400",
      "text-rose-400",
      "text-cyan-400",
      "text-fuchsia-400",
      "text-lime-400",
    ];
    const getColor = (n: number) => palette[(n - 1) % palette.length];

    return lines.map((raw, idx) => {
      const line = raw ?? "";
      const m = line.match(/^\s*(?:(Speaker\s+(\d+):)|(\[(\d+)\]:)|(Narrator:))/i);
      if (!m) {
        return (
          <div key={idx} className="whitespace-pre-wrap">
            {line}
          </div>
        );
      }
      let tag = "";
      let rest = line;
      let cls = "";
      if (m[1]) {
        // Speaker N:
        const n = parseInt(m[2] || "1", 10) || 1;
        tag = m[1];
        rest = line.slice(m[0].length);
        cls = getColor(n);
      } else if (m[3]) {
        // [N]:
        const n = parseInt(m[4] || "1", 10) || 1;
        tag = m[3];
        rest = line.slice(m[0].length);
        cls = getColor(n);
      } else if (m[5]) {
        // Narration: display as Speaker 1 for consistency in preview
        tag = "Speaker 1:";
        rest = line.slice(m[0].length);
        cls = getColor(1);
      }
      return (
        <div key={idx} className="whitespace-pre-wrap">
          <span className={`font-mono font-semibold ${cls}`}>{tag}</span>
          <span>{rest}</span>
        </div>
      );
    });
  }, [text]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      toast({
        title: "Copied to clipboard",
        description: "Text has been copied successfully",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Failed to copy text to clipboard",
        variant: "destructive",
      });
    }
  };

  const handleDownload = () => {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName
      ? `processed_${fileName.replace(/\.[^/.]+$/, "")}.txt`
      : "processed_text.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Download started",
      description: "File is being downloaded",
    });
  };

  return (
    <Card className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-2 p-4 border-b">
        <h3 className="text-base font-medium">Output</h3>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            disabled={!text}
            data-testid="button-copy-output"
          >
            <Copy className="h-4 w-4 mr-2" />
            Copy
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            disabled={!text}
            data-testid="button-download-output"
          >
            <Download className="h-4 w-4 mr-2" />
            Save
          </Button>
        </div>
      </div>
      <div className="flex-1 p-4 overflow-hidden">
        <div
          className="h-full w-full overflow-auto rounded-md border border-slate-800 bg-black text-white p-3 font-mono text-sm whitespace-pre-wrap"
          data-testid="textarea-output"
        >
          {coloredLines}
        </div>
      </div>
    </Card>
  );
}
