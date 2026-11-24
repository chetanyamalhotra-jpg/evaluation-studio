// Numeric, cost, hash, and timestamp formatters for Precision Telemetry UI.

export function formatUsd(
  value: number | null | undefined,
  options?: { compact?: boolean }
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  if (abs === 0) {
    return "$0.00";
  }
  const sign = value < 0 ? "-" : "";
  if (options?.compact && abs >= 1000) {
    return `${sign}$${(abs / 1000).toFixed(2)}k`;
  }
  if (abs < 0.0001) {
    return `${sign}$${abs.toFixed(6)}`;
  }
  if (abs < 0.01) {
    return `${sign}$${abs.toFixed(4)}`;
  }
  if (abs < 1) {
    return `${sign}$${abs.toFixed(3)}`;
  }
  return `${sign}$${abs.toFixed(2)}`;
}

export function formatMetric(
  value: number | null | undefined,
  mode: "percent" | "decimal" = "percent",
  decimals = 1
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  if (mode === "percent") {
    return `${(value * 100).toFixed(decimals)}%`;
  }
  return value.toFixed(decimals);
}

export function formatStderr(
  stderr: number | null | undefined,
  mode: "percent" | "decimal" = "percent",
  decimals = 1
): string {
  if (stderr === null || stderr === undefined || Number.isNaN(stderr)) {
    return "";
  }
  if (mode === "percent") {
    return `± ${(stderr * 100).toFixed(decimals)}%`;
  }
  return `± ${stderr.toFixed(decimals)}`;
}

export function shortHash(
  hash: string | null | undefined,
  head = 8,
  tail = 6
): string {
  if (!hash) {
    return "—";
  }
  if (hash.length <= head + tail + 1) {
    return hash;
  }
  return `${hash.slice(0, head)}…${hash.slice(-tail)}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return iso;
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

export function formatTokens(count: number | null | undefined): string {
  if (count === null || count === undefined || Number.isNaN(count)) {
    return "—";
  }
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 0,
  }).format(Math.round(count));
}

export function formatLatencyMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) {
    return "—";
  }
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) {
    return "—";
  }
  if (bytes < 1024) {
    return `${Math.round(bytes)} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
