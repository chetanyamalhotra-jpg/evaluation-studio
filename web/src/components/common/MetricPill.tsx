import { formatMetric, formatStderr } from "../../utils/format";
import { Badge, type BadgeVariant } from "./Badge";

export interface MetricPillProps {
  label: string;
  value: number | null | undefined;
  stderr?: number | null;
  unresolved?: boolean;
  mode?: "percent" | "decimal";
  decimals?: number;
  accent?: BadgeVariant;
  className?: string;
}

export function MetricPill({
  label,
  value,
  stderr,
  unresolved = false,
  mode = "percent",
  decimals = 1,
  accent = "cyan",
  className = "",
}: MetricPillProps) {
  const formattedVal = formatMetric(value, mode, decimals);
  const formattedErr = formatStderr(stderr, mode, decimals);

  return (
    <span className={`text-ui-sm inline-flex items-baseline gap-2 ${className}`}>
      <span className="font-medium" style={{ color: "var(--text-secondary)" }}>
        {label}
      </span>
      <span
        className="font-mono-num font-semibold"
        style={{
          color:
            unresolved
              ? "var(--accent-amber)"
              : accent === "neutral"
                ? "var(--text-primary)"
                : `var(--accent-${accent})`,
        }}
      >
        {formattedVal}
      </span>
      {formattedErr && (
        <span className="font-mono-num" style={{ color: "var(--text-muted)" }}>
          {formattedErr}
        </span>
      )}
      {unresolved && (
        <Badge variant="amber" size="sm" mono>
          UNRESOLVED
        </Badge>
      )}
    </span>
  );
}
