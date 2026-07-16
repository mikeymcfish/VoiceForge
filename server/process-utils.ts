import { spawn, type ChildProcess } from "child_process";

/** Best-effort termination for a spawned worker and its opted-in process group/tree. */
export function terminateChildProcess(
  child: ChildProcess,
  options: { processGroup?: boolean } = {}
): boolean {
  if (child.exitCode !== null || child.signalCode !== null) return false;

  if (process.platform === "win32" && child.pid) {
    try {
      let fallbackAttempted = false;
      const fallbackToDirectKill = () => {
        if (fallbackAttempted) return;
        fallbackAttempted = true;
        try {
          child.kill();
        } catch {}
      };
      const killer = spawn(
        "taskkill",
        ["/PID", String(child.pid), "/T", "/F"],
        { stdio: "ignore", windowsHide: true }
      );
      killer.once("error", fallbackToDirectKill);
      killer.once("close", (code) => {
        if (code !== 0) fallbackToDirectKill();
      });
      killer.unref();
      return true;
    } catch {
      // Fall through to Node's direct-process termination.
    }
  }

  if (process.platform !== "win32" && options.processGroup && child.pid) {
    try {
      process.kill(-child.pid, "SIGTERM");
      const forceTimer = setTimeout(() => {
        // The group may still contain a nested inference process after the
        // direct Python worker has exited, so always attempt the group kill.
        try {
          process.kill(-child.pid!, "SIGKILL");
        } catch {}
      }, 2_000);
      forceTimer.unref();
      return true;
    } catch {
      // Fall through to direct-process termination if the group no longer exists.
    }
  }

  try {
    const requested = child.kill("SIGTERM");
    if (requested) {
      const forceTimer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) {
          try {
            child.kill("SIGKILL");
          } catch {}
        }
      }, 2_000);
      forceTimer.unref();
      child.once("exit", () => clearTimeout(forceTimer));
    }
    return requested;
  } catch {
    return false;
  }
}

/**
 * Terminate a worker tree and wait for Windows taskkill to report success.
 * Unlike terminateChildProcess(), this intentionally has no direct-process
 * fallback on Windows: a surviving downloader descendant must block retry.
 */
export function terminateChildProcessTree(
  child: ChildProcess,
  options: { processGroup?: boolean } = {}
): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(false);
  if (process.platform !== "win32" || !child.pid) {
    return Promise.resolve(terminateChildProcess(child, options));
  }

  return new Promise<boolean>((resolve) => {
    try {
      const killer = spawn(
        "taskkill",
        ["/PID", String(child.pid), "/T", "/F"],
        { stdio: "ignore", windowsHide: true }
      );
      killer.once("error", () => resolve(false));
      killer.once("close", (code) => resolve(code === 0));
    } catch {
      resolve(false);
    }
  });
}
