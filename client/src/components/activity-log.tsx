import { AlertCircle, CheckCircle, Info, XCircle, Download, Trash2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { LogEntry } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

interface ActivityLogProps {
  logs: LogEntry[];
  onClear: () => void;
}

const logIcons = {
  info: Info,
  success: CheckCircle,
  warning: AlertCircle,
  error: XCircle,
};

const logColors = {
  info: "text-chart-4",
  success: "text-chart-2",
  warning: "text-chart-3",
  error: "text-destructive",
};

export function ActivityLog({ logs, onClear }: ActivityLogProps) {
  const { toast } = useToast();

  const handleExport = () => {
    const logText = logs
      .map(
        (log) =>
          `[${log.timestamp.toLocaleTimeString()}] ${log.type.toUpperCase()}: ${log.message}${log.details ? `\n  ${log.details}` : ""}`
      )
      .join("\n\n");

    const blob = new Blob([logText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `activity_log_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Log exported",
      description: "Activity log has been downloaded",
    });
  };

  return (
    <Card className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-2 p-4 border-b">
        <h3 className="text-base font-medium">Activity Log</h3>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleExport}
            disabled={logs.length === 0}
            data-testid="button-export-log"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            disabled={logs.length === 0}
            data-testid="button-clear-log"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Clear
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {logs.length === 0 ? (
            <div className="text-center py-8 text-sm text-muted-foreground">
              No activity logs yet
            </div>
          ) : (
            logs.map((log) => {
              const Icon = logIcons[log.type];
              return (
                <div
                  key={log.id}
                  className="flex items-start gap-3 text-sm"
                  data-testid={`log-entry-${log.type}`}
                >
                  <Icon className={`h-4 w-4 mt-0.5 ${logColors[log.type]}`} />
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="flex items-baseline gap-2">
                      <span className="text-xs text-muted-foreground font-mono">
                        {log.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm">{log.message}</p>
                    {log.details && (
                      <p className="text-xs text-muted-foreground font-mono">
                        {log.details}
                      </p>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}
