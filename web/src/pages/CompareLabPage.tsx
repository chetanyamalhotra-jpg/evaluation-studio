import { BarChart3, CheckSquare, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import {
  CategoryBreakdownTable,
  GroupedBarErrorChart,
  ParetoScatterChart,
  RadarProfileChart,
} from "../components/charts/AnalyticsCharts";
import { Badge, Card, EmptyState, ErrorBanner, LoadingSkeleton } from "../components/common";
import type { CompareResponse, ReceiptListItem } from "../types/evalStudio";
import { formatMetric, formatUsd } from "../utils/format";

export function CompareLabPage() {
  const [receipts, setReceipts] = useState<ReceiptListItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [baselineId, setBaselineId] = useState<string>("");
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .listReceipts(controller.signal)
      .then((res) => {
        setReceipts(res.receipts);
        if (res.receipts.length >= 2) {
          const initial = res.receipts.slice(0, 2).map((r) => r.dir_name);
          setSelectedIds(initial);
          setBaselineId(initial[0] ?? "");
        } else if (res.receipts.length === 1) {
          const only = res.receipts[0]!.dir_name;
          setSelectedIds([only]);
          setBaselineId(only);
        }
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name !== "AbortError") setError(err.message);
      })
      .finally(() => setLoadingList(false));
    return () => controller.abort();
  }, []);

  const toggleReceipt = (id: string) => {
    setSelectedIds((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
      if (!next.includes(baselineId)) {
        setBaselineId(next[0] ?? "");
      }
      return next;
    });
  };

  const runComparison = async () => {
    if (selectedIds.length === 0) return;
    setComparing(true);
    setError(null);
    try {
      const res = await api.compareReceipts({
        receipt_ids: selectedIds,
        baseline_id: baselineId || selectedIds[0],
        bootstrap_iters: 1000,
      });
      setComparison(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Comparison failed.");
    } finally {
      setComparing(false);
    }
  };

  const primaryMetric =
    comparison?.runs[0] ? Object.keys(comparison.runs[0].metrics)[0] ?? "exact_match" : "exact_match";

  return (
    <div className="space-y-6">
      {error && <ErrorBanner error={error} />}

      <Card
        title="Select Receipts to Compare"
        subtitle="Pick 2+ receipts to compute 95% CIs, paired bootstrap deltas, McNemar's test, and item-level diffs"
        actions={
          <button
            type="button"
            disabled={selectedIds.length === 0 || comparing}
            onClick={() => void runComparison()}
            className="rounded-lg px-4 py-1.5 text-xs font-semibold transition-opacity disabled:opacity-40"
            style={{ backgroundColor: "var(--accent-cyan)", color: "#090D16" }}
          >
            {comparing ? "Computing Stats..." : `Compare (${selectedIds.length})`}
          </button>
        }
      >
        {loadingList ? (
          <LoadingSkeleton lines={3} />
        ) : receipts.length === 0 ? (
          <EmptyState
            icon={<BarChart3 className="h-8 w-8" />}
            title="No receipts available to compare"
            description="Run at least two evaluations first so you can compare models statistically."
          />
        ) : (
          <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2 lg:grid-cols-3">
            {receipts.map((r) => {
              const checked = selectedIds.includes(r.dir_name);
              const isBase = baselineId === r.dir_name;
              return (
                <div
                  key={r.dir_name}
                  className="flex items-start justify-between gap-2 rounded-xl border p-3 transition-colors"
                  style={{
                    borderColor: checked ? "var(--accent-cyan)" : "var(--border-subtle)",
                    backgroundColor: checked ? "var(--accent-cyan-soft)" : "var(--bg-elevated)",
                  }}
                >
                  <button
                    type="button"
                    onClick={() => toggleReceipt(r.dir_name)}
                    className="flex min-w-0 flex-1 items-start gap-2.5 text-left"
                  >
                    {checked ? (
                      <CheckSquare className="mt-0.5 h-4 w-4 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                    ) : (
                      <Square className="mt-0.5 h-4 w-4 shrink-0" style={{ color: "var(--text-muted)" }} />
                    )}
                    <div className="min-w-0">
                      <div className="truncate text-xs font-semibold">{r.model_id}</div>
                      <div className="truncate text-[11px]" style={{ color: "var(--text-secondary)" }}>
                        {r.task_name} · {formatUsd(r.total_cost_usd)}
                      </div>
                    </div>
                  </button>
                  {checked && (
                    <button
                      type="button"
                      onClick={() => setBaselineId(r.dir_name)}
                      className="shrink-0 rounded-md px-2 py-0.5 text-[10px] font-semibold"
                      style={{
                        backgroundColor: isBase ? "var(--accent-emerald)" : "var(--bg-surface)",
                        color: isBase ? "#090D16" : "var(--text-secondary)",
                      }}
                    >
                      {isBase ? "Baseline" : "Set Base"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </Card>

      {comparison && (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <GroupedBarErrorChart runs={comparison.runs} metricKey={primaryMetric} />
            <ParetoScatterChart runs={comparison.runs} metricKey={primaryMetric} />
          </div>

          <RadarProfileChart runs={comparison.runs} />

          <CategoryBreakdownTable
            runs={comparison.runs}
            entries={comparison.category_breakdown}
          />

          {/* Pairwise Statistical Matrix & Item Diffs */}
          {comparison.pairwise.map((pair) => (
            <Card
              key={`${pair.baseline_id}-${pair.challenger_id}`}
              title={`Head-to-Head: ${pair.challenger_id} vs. ${pair.baseline_id}`}
              subtitle={`Metric: ${pair.metric}`}
              actions={
                <Badge variant={pair.unresolved ? "amber" : "emerald"}>
                  {pair.unresolved ? "UNRESOLVED (CI spans 0)" : "SIGNIFICANT"}
                </Badge>
              }
            >
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg border p-3" style={{ borderColor: "var(--border-subtle)" }}>
                  <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    Paired Bootstrap Δ
                  </div>
                  <div className="mt-1 font-mono text-sm font-semibold tabular-nums">
                    {(pair.paired_bootstrap.mean_delta * 100).toFixed(2)}%
                  </div>
                  <div className="font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>
                    p = {pair.paired_bootstrap.p_value.toFixed(4)}
                  </div>
                </div>
                <div className="rounded-lg border p-3" style={{ borderColor: "var(--border-subtle)" }}>
                  <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    95% Bootstrap CI
                  </div>
                  <div className="mt-1 font-mono text-sm font-semibold tabular-nums">
                    [{(pair.paired_bootstrap.ci_low * 100).toFixed(1)}%,{" "}
                    {(pair.paired_bootstrap.ci_high * 100).toFixed(1)}%]
                  </div>
                </div>
                <div className="rounded-lg border p-3" style={{ borderColor: "var(--border-subtle)" }}>
                  <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    McNemar Test
                  </div>
                  <div className="mt-1 font-mono text-sm font-semibold tabular-nums">
                    {pair.mcnemar ? `p = ${pair.mcnemar.p_value.toFixed(4)}` : "N/A (non-binary)"}
                  </div>
                  {pair.mcnemar && (
                    <div className="font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>
                      n01={pair.mcnemar.n_01} · n10={pair.mcnemar.n_10}
                    </div>
                  )}
                </div>
                <div className="rounded-lg border p-3" style={{ borderColor: "var(--border-subtle)" }}>
                  <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    Decision Flip Rate
                  </div>
                  <div className="mt-1 font-mono text-sm font-semibold tabular-nums">
                    {pair.noise_floor ? formatMetric(pair.noise_floor.flip_rate) : "—"}
                  </div>
                  {pair.noise_floor && (
                    <div className="font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>
                      {pair.noise_floor.n_flips} / {pair.noise_floor.n_decisions} flips
                    </div>
                  )}
                </div>
              </div>

              {pair.item_diffs.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                        <th className="py-2 pr-3">Sample ID</th>
                        <th className="py-2 pr-3">Baseline</th>
                        <th className="py-2 pr-3">Challenger</th>
                        <th className="py-2">Target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pair.item_diffs.slice(0, 20).map((diff) => (
                        <tr
                          key={diff.sample_id}
                          className="border-b last:border-b-0"
                          style={{ borderColor: "var(--border-subtle)" }}
                        >
                          <td className="py-2 pr-3 font-mono text-[11px]">{diff.sample_id}</td>
                          <td className="py-2 pr-3 font-mono">
                            {diff.baseline_score} · {diff.baseline_output.slice(0, 36)}
                          </td>
                          <td className="py-2 pr-3 font-mono">
                            {diff.challenger_score} · {diff.challenger_output.slice(0, 36)}
                          </td>
                          <td className="py-2 font-mono text-[11px]" style={{ color: "var(--text-secondary)" }}>
                            {String(diff.target).slice(0, 36)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  Zero item-level score differences between baseline and challenger.
                </div>
              )}
            </Card>
          ))}
        </>
      )}
    </div>
  );
}
