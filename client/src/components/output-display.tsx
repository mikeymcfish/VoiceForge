import { useMemo } from "react";
import { Copy, Download, FilePenLine, ScanText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { countWords } from "@shared/text-utils";

interface OutputDisplayProps {
  text: string;
  sourceText?: string;
  fileName?: string;
  onChange?: (value: string) => void;
}

const speakerPalette = [
  "text-violet-600 dark:text-violet-300",
  "text-cyan-700 dark:text-cyan-300",
  "text-emerald-700 dark:text-emerald-300",
  "text-amber-700 dark:text-amber-300",
  "text-rose-700 dark:text-rose-300",
  "text-blue-700 dark:text-blue-300",
];

export function OutputDisplay({ text, sourceText = "", fileName, onChange }: OutputDisplayProps) {
  const { toast } = useToast();

  const coloredLines = useMemo(() => {
    return (text || "").split(/\r?\n/).map((line, index) => {
      const match = line.match(/^\s*(?:(Speaker\s+(\d+):)|(\[(\d+)\]:)|(Narrator:))/i);
      if (!match) return <div key={index} className="min-h-[1.5em] whitespace-pre-wrap">{line}</div>;

      const matchedTag = match[1] || match[3] || match[5] || "";
      const speakerNumber = Number(match[2] || match[4] || 0);
      const colorClass = match[5]
        ? "text-fuchsia-700 dark:text-fuchsia-300"
        : speakerPalette[Math.max(0, speakerNumber - 1) % speakerPalette.length];

      return (
        <div key={index} className="min-h-[1.5em] whitespace-pre-wrap">
          <span className={`font-mono text-[0.92em] font-semibold ${colorClass}`}>{matchedTag}</span>
          <span>{line.slice(match[0].length)}</span>
        </div>
      );
    });
  }, [text]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      toast({ title: "Copied", description: "The reviewed result is on your clipboard." });
    } catch {
      toast({ title: "Copy failed", description: "Your browser blocked clipboard access.", variant: "destructive" });
    }
  };

  const handleDownload = () => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName
      ? `${fileName.replace(/\.[^/.]+$/, "")}_voiceforge.txt`
      : "voiceforge_script.txt";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    toast({ title: "Script saved", description: "Your reviewed text was downloaded." });
  };

  return (
    <Card className="flex h-full min-h-[620px] flex-col overflow-hidden rounded-2xl border-card-border shadow-sm">
      <Tabs defaultValue="edit" className="flex min-h-0 flex-1 flex-col">
        <div className="flex flex-col gap-3 border-b px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-primary/10 text-primary">
              <FilePenLine className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold">Script review</h3>
                {text && <Badge variant="outline" className="rounded-full text-[10px]">{countWords(text).toLocaleString()} words</Badge>}
              </div>
              <p className="mt-0.5 text-[11px] text-muted-foreground">Preview, correct, and compare before export.</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleCopy} disabled={!text} className="rounded-lg">
              <Copy className="mr-1.5 h-3.5 w-3.5" /> Copy
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload} disabled={!text} className="rounded-lg">
              <Download className="mr-1.5 h-3.5 w-3.5" /> Save
            </Button>
          </div>
        </div>

        <div className="border-b px-4 py-2 sm:px-5">
          <TabsList className="h-9 rounded-lg bg-muted/70 p-1">
            <TabsTrigger value="edit" className="h-7 rounded-md px-3 text-xs">Edit result</TabsTrigger>
            <TabsTrigger value="preview" className="h-7 rounded-md px-3 text-xs">Speaker preview</TabsTrigger>
            <TabsTrigger value="source" className="h-7 rounded-md px-3 text-xs">Original source</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="edit" className="m-0 min-h-0 flex-1 p-4 sm:p-5">
          {text ? (
            <Textarea
              value={text}
              onChange={(event) => onChange?.(event.target.value)}
              readOnly={!onChange}
              className="h-full min-h-[470px] resize-none rounded-xl border-border/80 bg-background/55 p-4 font-mono text-[13px] leading-6"
              data-testid="textarea-output"
              aria-label="Editable processed script"
            />
          ) : (
            <EmptyResult />
          )}
        </TabsContent>

        <TabsContent value="preview" className="m-0 min-h-0 flex-1 p-4 sm:p-5">
          {text ? (
            <div className="h-full min-h-[470px] overflow-auto rounded-xl border bg-background/55 p-4 text-sm leading-7" data-testid="preview-output">
              {coloredLines}
            </div>
          ) : (
            <EmptyResult />
          )}
        </TabsContent>

        <TabsContent value="source" className="m-0 min-h-0 flex-1 p-4 sm:p-5">
          {sourceText ? (
            <Textarea
              value={sourceText}
              readOnly
              className="h-full min-h-[470px] resize-none rounded-xl border-border/80 bg-muted/25 p-4 text-[13px] leading-6 text-muted-foreground"
              aria-label="Original source text"
            />
          ) : (
            <EmptyResult source />
          )}
        </TabsContent>
      </Tabs>
    </Card>
  );
}

function EmptyResult({ source = false }: { source?: boolean }) {
  return (
    <div className="grid h-full min-h-[470px] place-items-center rounded-xl border border-dashed bg-muted/15 p-8 text-center">
      <div className="max-w-xs">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-primary/10 text-primary">
          <ScanText className="h-5 w-5" />
        </div>
        <p className="mt-4 text-sm font-bold">{source ? "Add a source to begin" : "Your result will appear here"}</p>
        <p className="mt-1.5 text-xs leading-5 text-muted-foreground">
          {source ? "Upload a TXT or EPUB file, or paste directly into the source editor." : "Try safe cleanup for a local preview, or run the selected AI workflow."}
        </p>
      </div>
    </div>
  );
}
