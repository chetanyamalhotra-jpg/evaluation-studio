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
  title = "Request failed",
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
      className={`animate-fade-slide-up text-ui-base flex items-start gap-3 rounded-md border px-4 py-3 ${className}`}
      style={{
        backgroundColor: "var(--accent-rose-soft)",
        borderColor: "rgba(244, 63, 94, 0.4)",
        color: "var(--text-primary)",
      }}
    >
      <AlertTriangle
        className="mt-0.5 h-4 w-4 shrink-0"
        style={{ color: "var(--accent-rose)" }}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2.5">
          <p
            className="text-ui-sm font-semibold"
            style={{ color: "var(--accent-rose)" }}
          >
            {title}
          </p>
          {error instanceof ApiError && (
            <span
              className="font-mono-num text-ui-sm"
              style={{ color: "var(--accent-rose)" }}
            >
              HTTP {error.status}
            </span>
          )}
        </div>
        <p
          className="text-ui-sm mt-1 break-words"
          style={{ color: "var(--text-primary)" }}
        >
          {message}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1 font-medium"
            style={{
              backgroundColor: "var(--bg-surface)",
              borderColor: "var(--border-subtle)",
              color: "var(--text-primary)",
            }}
          >
            <RotateCw className="h-3.5 w-3.5" />
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Dismiss error"
            className="telemetry-transition cursor-pointer rounded-md p-1 hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}
