import { AlertTriangle, RotateCw, X } from "lucide-react";
import { ApiError } from "../../api/client";

export interface ErrorBannerProps {
  title?: string;
  error: string | Error | null | undefined;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export function ErrorBanner({
  title = "Request Failed",
  error,
  onRetry,
  onDismiss,
  className = "",
}: ErrorBannerProps) {
  if (!error) return null;

  const message =
    error instanceof ApiError
      ? error.detail
      : error instanceof Error
        ? error.message
        : String(error);

  return (
    <div
      role="alert"
      className={`animate-fade-slide-up flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${className}`}
      style={{
        backgroundColor: "var(--accent-rose-soft)",
        borderColor: "rgba(244, 63, 94, 0.35)",
        color: "var(--text-primary)",
      }}
    >
      <AlertTriangle
        className="mt-0.5 h-4 w-4 shrink-0"
        style={{ color: "var(--accent-rose)" }}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p
            className="text-xs font-semibold tracking-tight"
            style={{ color: "var(--accent-rose)" }}
          >
            {title}
          </p>
          {error instanceof ApiError && (
            <span
              className="font-mono-num rounded px-1.5 py-0.5 text-[10px]"
              style={{
                backgroundColor: "rgba(244, 63, 94, 0.16)",
                color: "var(--accent-rose)",
              }}
            >
              HTTP {error.status}
            </span>
          )}
        </div>
        <p
          className="mt-1 break-words text-xs leading-relaxed"
          style={{ color: "var(--text-primary)" }}
        >
          {message}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="telemetry-transition inline-flex cursor-pointer items-center gap-1 rounded-lg border px-2.5 py-1 text-xs font-medium"
            style={{
              backgroundColor: "var(--bg-surface)",
              borderColor: "var(--border-subtle)",
              color: "var(--text-primary)",
            }}
          >
            <RotateCw className="h-3 w-3" />
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Dismiss error"
            className="telemetry-transition cursor-pointer rounded-lg p-1 hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}
