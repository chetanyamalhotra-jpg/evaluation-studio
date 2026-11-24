import { Download } from "lucide-react";
import { useRef } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { CategoryBreakdownEntry, CompareRunSummary } from "../../types/evalStudio";
import { downloadCsv, exportElementToPng, exportElementToSvg } from "../../utils/export";
import { formatMetric, formatUsd } from "../../utils/format";
import { Card } from "../common";

const SERIES_COLORS = [
  "#06B6D4", // Cyan
  "#10B981", // Emerald
  "#8B5CF6", // Violet
  "#F59E0B", // Amber
  "#F43F5E", // Rose
  "#38BDF8", // Sky
];

const TOOLTIP_STYLE = {
  backgroundColor: "var(--bg-elevated)",
  border: "1px solid var(--border-subtle)",
  borderRadius: "10px",
  color: "var(--text-primary)",
  fontSize: "12px",
};

function ExportToolbar({
  containerRef,
  basename,
  csvRows,
}: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  basename: string;
  csvRows: Record<string, unknown>[];
}) {
  return (
    <div className="flex items-center gap-1.5">
      <button
        type="button"
        onClick={() => {
          if (containerRef.current) void exportElementToPng(containerRef.current, basename);
        }}
        className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium transition-colors hover:opacity-85"
        style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
      >
        <Download className="h-3 w-3" /> PNG
      </button>
      <button
        type="button"
        onClick={() => {
          if (containerRef.current) exportElementToSvg(containerRef.current, basename);
        }}
        className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium transition-colors hover:opacity-85"
        style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
      >
        SVG
      </button>
      <button
        type="button"
        onClick={() => downloadCsv(csvRows, basename)}
        className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium transition-colors hover:opacity-85"
        style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
      >
        CSV
      </button>
    </div>
  );
}

export function GroupedBarErrorChart({
  runs,
  metricKey,
}: {
  runs: CompareRunSummary[];
  metricKey: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const data = runs.map((run) => {
    const m = run.metrics[metricKey];
    const value = m ? Number((m.value * 100).toFixed(2)) : 0;
    const halfCi = m && m.stderr ? Number((m.stderr * 1.96 * 100).toFixed(2)) : 0;
    return {
      label: `${run.model_id} (${run.task_name})`,
      model: run.model_id,
      score: value,
      error: halfCi,
      unresolved: m?.unresolved ?? false,
    };
  });

  return (
    <Card
      title={`Accuracy & 95% CI (${metricKey})`}
      subtitle="Bars show mean score (%) with ± 1.96 × stderr confidence intervals"
      actions={<ExportToolbar containerRef={ref} basename={`eval-studio-bar-${metricKey}`} csvRows={data} />}
    >
      <div ref={ref} className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 20, left: 0, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis dataKey="model" tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} unit="%" />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Bar dataKey="score" name={metricKey} fill={SERIES_COLORS[0]} radius={[6, 6, 0, 0]}>
              <ErrorBar dataKey="error" width={6} strokeWidth={2} stroke="#F59E0B" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function RadarProfileChart({ runs }: { runs: CompareRunSummary[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const metricKeys = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.metrics)))
  ).slice(0, 6);

  const radarData = metricKeys.map((metric) => {
    const row: Record<string, string | number> = { metric };
    for (const run of runs) {
      const m = run.metrics[metric];
      row[run.receipt_id] = m ? Number((m.value * 100).toFixed(1)) : 0;
    }
    return row;
  });

  return (
    <Card
      title="Multi-Metric Model Profile"
      subtitle="Normalized radar profile across evaluated metrics"
      actions={<ExportToolbar containerRef={ref} basename="eval-studio-radar" csvRows={radarData} />}
    >
      <div ref={ref} className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="var(--border-subtle)" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--text-secondary)", fontSize: 11 }} />
            <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Legend />
            {runs.slice(0, 5).map((run, idx) => (
              <Radar
                key={run.receipt_id}
                name={run.model_id}
                dataKey={run.receipt_id}
                stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                fill={SERIES_COLORS[idx % SERIES_COLORS.length]}
                fillOpacity={0.22}
              />
            ))}
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function ParetoScatterChart({
  runs,
  metricKey,
}: {
  runs: CompareRunSummary[];
  metricKey: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const points = runs.map((run) => ({
    model: run.model_id,
    cost_usd: Number(run.total_cost_usd.toFixed(6)),
    score_pct: Number(((run.metrics[metricKey]?.value ?? 0) * 100).toFixed(2)),
    tokens: run.total_prompt_tokens + run.total_completion_tokens,
  }));

  return (
    <Card
      title="Cost vs. Accuracy Pareto Frontier"
      subtitle="Total USD spend vs. primary metric score (bubble size = total tokens)"
      actions={<ExportToolbar containerRef={ref} basename="eval-studio-pareto" csvRows={points} />}
    >
      <div ref={ref} className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 16, right: 24, bottom: 16, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis
              type="number"
              dataKey="cost_usd"
              name="Cost (USD)"
              tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
              tickFormatter={(v: number) => formatUsd(v)}
            />
            <YAxis
              type="number"
              dataKey="score_pct"
              name="Score (%)"
              domain={[0, 100]}
              unit="%"
              tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
            />
            <ZAxis type="number" dataKey="tokens" range={[80, 420]} name="Tokens" />
            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3" }} />
            <Scatter name="Runs" data={points} fill={SERIES_COLORS[1]} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function CategoryBreakdownTable({
  runs,
  entries,
}: {
  runs: CompareRunSummary[];
  entries: CategoryBreakdownEntry[];
}) {
  if (entries.length === 0) return null;

  return (
    <Card
      title="Paper & Category Breakdown Heatmap"
      subtitle="Per-category mean accuracy across selected runs"
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
              <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                Task / Group
              </th>
              <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                Category
              </th>
              {runs.map((run) => (
                <th
                  key={run.receipt_id}
                  className="py-2 px-3 text-right font-semibold"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {run.model_id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr
                key={`${entry.group}-${entry.category}`}
                className="border-b last:border-b-0"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <td className="py-2.5 pr-4 font-medium">{entry.group}</td>
                <td className="py-2.5 pr-4 font-mono text-[11px]" style={{ color: "var(--accent-cyan)" }}>
                  {entry.category}
                </td>
                {runs.map((run) => {
                  const val = entry.scores[run.receipt_id];
                  const pct = val !== undefined ? val * 100 : null;
                  const bg =
                    pct === null
                      ? "transparent"
                      : pct >= 80
                        ? "var(--accent-emerald-soft)"
                        : pct >= 50
                          ? "var(--accent-amber-soft)"
                          : "var(--accent-rose-soft)";
                  return (
                    <td key={run.receipt_id} className="py-2 px-3 text-right">
                      <span
                        className="inline-block rounded-md px-2 py-0.5 font-mono tabular-nums"
                        style={{ backgroundColor: bg }}
                      >
                        {val !== undefined ? formatMetric(val) : "—"}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
