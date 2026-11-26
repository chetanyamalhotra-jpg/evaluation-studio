import { useRef } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  "#06B6D4", // Cyan primary
  "#10B981", // Emerald
  "#38BDF8", // Sky
  "#F59E0B", // Amber
  "#F43F5E", // Rose
  "#14B8A6", // Teal
];

const TOOLTIP_STYLE = {
  backgroundColor: "var(--bg-elevated)",
  border: "1px solid var(--border-subtle)",
  borderRadius: "6px",
  color: "var(--text-primary)",
  fontSize: "13px",
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
    <div
      className="inline-flex divide-x overflow-hidden rounded-md border"
      style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
    >
      <button
        type="button"
        onClick={() => {
          if (containerRef.current) void exportElementToPng(containerRef.current, basename);
        }}
        className="telemetry-transition text-ui-sm cursor-pointer px-2.5 py-1 font-medium hover:opacity-80"
        style={{ color: "var(--text-secondary)" }}
      >
        PNG
      </button>
      <button
        type="button"
        onClick={() => {
          if (containerRef.current) exportElementToSvg(containerRef.current, basename);
        }}
        className="telemetry-transition text-ui-sm cursor-pointer px-2.5 py-1 font-medium hover:opacity-80"
        style={{ color: "var(--text-secondary)", borderColor: "var(--border-subtle)" }}
      >
        SVG
      </button>
      <button
        type="button"
        onClick={() => downloadCsv(csvRows, basename)}
        className="telemetry-transition text-ui-sm cursor-pointer px-2.5 py-1 font-medium hover:opacity-80"
        style={{ color: "var(--text-secondary)", borderColor: "var(--border-subtle)" }}
      >
        CSV
      </button>
    </div>
  );
}

interface BarDatum {
  label: string;
  model: string;
  score: number;
  error: number;
  unresolved: boolean;
}

function GroupedBarTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: ReadonlyArray<{ payload: BarDatum }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const item = payload[0]!.payload;
  return (
    <div className="p-3" style={TOOLTIP_STYLE}>
      <div className="font-semibold">{item.label}</div>
      <div className="font-mono-num text-ui-sm mt-1">
        Score: {item.score.toFixed(1)}% ± {item.error.toFixed(1)}% (95% CI)
      </div>
      {item.unresolved && (
        <div
          className="font-mono-num text-ui-sm mt-1 font-semibold"
          style={{ color: "var(--accent-amber)" }}
        >
          UNRESOLVED (cross-set or CI overlap)
        </div>
      )}
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

  const data: BarDatum[] = runs.map((run) => {
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
      title={`Accuracy and 95% confidence interval (${metricKey})`}
      subtitle="Cyan bars indicate resolved scores; amber dashed bars indicate UNRESOLVED intervals"
      actions={
        <ExportToolbar
          containerRef={ref}
          basename={`eval-studio-bar-${metricKey}`}
          csvRows={data as unknown as Record<string, unknown>[]}
        />
      }
    >
      <div ref={ref} className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 20, left: 0, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis dataKey="model" tick={{ fill: "var(--text-secondary)", fontSize: 13 }} />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "var(--text-secondary)", fontSize: 13 }}
              unit="%"
            />
            <Tooltip content={<GroupedBarTooltip />} />
            <Bar dataKey="score" name={metricKey} radius={[4, 4, 0, 0]}>
              {data.map((entry, idx) => (
                <Cell
                  key={`cell-${idx}`}
                  fill={entry.unresolved ? "#F59E0B" : SERIES_COLORS[0]}
                  stroke={entry.unresolved ? "#F59E0B" : "none"}
                  strokeDasharray={entry.unresolved ? "4 2" : undefined}
                />
              ))}
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
      title="Multi-metric model profile"
      subtitle="Normalized radar comparison across evaluated metrics"
      actions={<ExportToolbar containerRef={ref} basename="eval-studio-radar" csvRows={radarData} />}
    >
      <div ref={ref} className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="var(--border-subtle)" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--text-secondary)", fontSize: 13 }} />
            <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 13 }} />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: "13px" }} />
            {runs.slice(0, 5).map((run, idx) => (
              <Radar
                key={run.receipt_id}
                name={run.model_id}
                dataKey={run.receipt_id}
                stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                fill={SERIES_COLORS[idx % SERIES_COLORS.length]}
                fillOpacity={0.2}
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
      title="Cost versus accuracy Pareto frontier"
      subtitle="Total USD spend versus primary metric score (point area reflects token volume)"
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
              tick={{ fill: "var(--text-secondary)", fontSize: 13 }}
              tickFormatter={(v: number) => formatUsd(v)}
            />
            <YAxis
              type="number"
              dataKey="score_pct"
              name="Score (%)"
              domain={[0, 100]}
              unit="%"
              tick={{ fill: "var(--text-secondary)", fontSize: 13 }}
            />
            <ZAxis type="number" dataKey="tokens" range={[90, 420]} name="Tokens" />
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
      title="Paper and category breakdown"
      subtitle="Per-category mean accuracy across compared runs"
      padding="none"
    >
      <div className="overflow-x-auto">
        <table className="text-ui-sm w-full text-left">
          <thead>
            <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
              <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                Task / Group
              </th>
              <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                Category
              </th>
              {runs.map((run) => (
                <th
                  key={run.receipt_id}
                  className="px-5 py-3 text-right font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {run.model_id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
            {entries.map((entry) => (
              <tr key={`${entry.group}-${entry.category}`}>
                <td className="px-5 py-3 font-medium">{entry.group}</td>
                <td
                  className="font-mono-num px-4 py-3"
                  style={{ color: "var(--accent-cyan)" }}
                >
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
                    <td
                      key={run.receipt_id}
                      className="font-mono-num px-5 py-3 text-right font-medium"
                      style={{ backgroundColor: bg }}
                    >
                      {val !== undefined ? formatMetric(val) : "--"}
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
