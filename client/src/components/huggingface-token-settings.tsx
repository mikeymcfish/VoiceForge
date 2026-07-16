import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import type { HuggingFaceTokenStatus } from "@shared/schema";
import { useQueryClient } from "@tanstack/react-query";

interface HuggingFaceTokenSettingsProps {
  disabled?: boolean;
}

export function HuggingFaceTokenSettings({ disabled }: HuggingFaceTokenSettingsProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<HuggingFaceTokenStatus | null>(null);
  const [tokenInput, setTokenInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/settings/huggingface-token");
      if (!res.ok) throw new Error("Failed to load token status");
      const data: HuggingFaceTokenStatus = await res.json();
      setStatus(data);
    } catch (error) {
      console.error("Failed to load HuggingFace token status:", error);
      toast({
        title: "Unable to load token status",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void loadStatus();
  }, [loadStatus]);

  const handleSave = async () => {
    if (!tokenInput.trim()) {
      toast({
        title: "Enter a token",
        description: "Paste your Hugging Face API token before saving.",
        variant: "destructive",
      });
      return;
    }
    setSaving(true);
    try {
      const res = await fetch("/api/settings/huggingface-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: tokenInput }),
      });
      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg?.error || "Failed to save API token");
      }
      const data: HuggingFaceTokenStatus = await res.json();
      setStatus(data);
      setTokenInput("");
      void queryClient.invalidateQueries({ queryKey: ["huggingface-usage"] });
      void queryClient.invalidateQueries({ queryKey: ["speech-status"] });
      toast({
        title: "Token saved",
        description: "Your Hugging Face API token is now active.",
      });
    } catch (error) {
      console.error("Failed to save HuggingFace token:", error);
      toast({
        title: "Save failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    setSaving(true);
    try {
      const res = await fetch("/api/settings/huggingface-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: null }),
      });
      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg?.error || "Failed to remove API token");
      }
      const data: HuggingFaceTokenStatus = await res.json();
      setStatus(data);
      setTokenInput("");
      void queryClient.invalidateQueries({ queryKey: ["huggingface-usage"] });
      void queryClient.invalidateQueries({ queryKey: ["speech-status"] });
      toast({
        title: "Token cleared",
        description: "API mode will require a new token before use.",
      });
    } catch (error) {
      console.error("Failed to clear HuggingFace token:", error);
      toast({
        title: "Clear failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const busy = disabled || saving;

  return (
    <div className="mt-3 space-y-2">
      <div className="space-y-1">
        <Label htmlFor="hf-token" className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Hugging Face API token
        </Label>
        <Input
          id="hf-token"
          type="password"
          value={tokenInput}
          onChange={(event) => setTokenInput(event.target.value)}
          placeholder="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
          disabled={busy}
          autoComplete="off"
          className="h-9"
        />
      </div>

      <div className="flex items-center gap-2">
        <Button size="sm" onClick={handleSave} disabled={busy}>
          Save Token
        </Button>
        <Button size="sm" variant="secondary" onClick={handleClear} disabled={busy || !status?.configured}>
          Clear Token
        </Button>
        <Button size="sm" variant="ghost" onClick={loadStatus} disabled={busy || loading}>
          Refresh
        </Button>
      </div>

      <p className="text-xs text-muted-foreground">
        {loading
          ? "Checking current token status…"
          : status?.configured
            ? `Token configured (${status.tokenPreview ?? "hidden"})`
            : "No Hugging Face API token configured. Paste one above to enable API mode."}
      </p>
      <p className="text-[11px] leading-4 text-muted-foreground">
        A standard read token is recommended for public Space calls and account-usage bars. Inference-only fine-grained tokens may not expose ZeroGPU or billing usage.
      </p>
    </div>
  );
}
