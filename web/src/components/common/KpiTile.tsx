import type { ReactNode } from "react";

export type AccentTone = "cyan" | "emerald" | "amber" | "rose";

export interface KpiTileProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  delta?: string;
  accent?: AccentTone;
  icon?: ReactNode;
  loading?: boolean;
  className?: string;
}

const ACCENT_COLOR: Record<AccentTone, { fg: string }> = {
  cyan: { fg: "var(--accent-cyan)" },
  emerald: { fg: "var(--accent-emerald)" },
  amber: { fg: "var(--accent-amber)" },
  rose: { fg: "var(--accent-rose)" },
};

export function KpiTile({
  label,
  value,
  hint,
  delta,
  accent = "cyan",
  icon,
  loading = false,
  className = "",
}: KpiTileProps) {
  const tone = ACCENT_COLOR[accent];

  return (
    <div
      className={`telemetry-transition relative p-5 ${className}`}
      style={{ backgroundColor: "var(--bg-surface)" }}
    >
      <div className="flex items-center justify-between gap-3">
        <span
          className="text-ui-sm font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          {label}
        </span>
        {icon && (
          <span className="shrink-0" style={{ color: tone.fg }}>
            {icon}
          </span>
        )}
      </div>

      <div className="mt-2.5 flex items-baseline gap-2.5">
        {loading ? (
          <div
            className="h-6 w-24 animate-pulse rounded"
            style={{ backgroundColor: "var(--bg-elevated)" }}
          />
        ) : (
          <div
            className="font-mono-num text-ui-lg font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            {value}
          </div>
        )}
        {delta && !loading && (
          <span
            className="font-mono-num text-ui-sm font-medium"
            style={{ color: tone.fg }}
          >
            {delta}
          </span>
        )}
      </div>

      {hint && (
        <p
          className="text-ui-sm mt-1 truncate"
          style={{ color: "var(--text-muted)" }}
        >
          {hint}
        </p>
      )}
    </div>
  );
}
