import { BarChart3 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import {
  CategoryBreakdownTable,
  GroupedBarErrorChart,
  ParetoScatterChart,
  RadarProfileChart,
} from "../components/charts/AnalyticsCharts";
import {
  Card,
  EmptyState,
  ErrorBanner,
  LoadingSkeleton,
  StatusBadge,
} from "../components/common";
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
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1
            className="text-[26px] font-semibold tracking-[-0.03em]"
            style={{ color: "var(--text-primary)" }}
          >
            Statistical model comparison
          </h1>
          <p className="text-ui-base mt-1 max-w-[64ch]" style={{ color: "var(--text-secondary)" }}>
            Compare receipts with 95% confidence intervals, paired bootstrap deltas, McNemar tests, and item-level diffs.
          </p>
        </div>
        <button
          type="button"
          disabled={selectedIds.length === 0 || comparing}
          onClick={() => void runComparison()}
          className="telemetry-transition text-ui-base cursor-pointer rounded-md px-4 py-2 font-semibold hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
        >
          {comparing ? "Computing statistics..." : `Compare selected (${selectedIds.length})`}
        </button>
      </div>

      {error && <ErrorBanner error={error} />}

      <Card
        title="Select receipts and baseline"
        subtitle="Check two or more runs to compare and designate one baseline reference"
        padding="none"
      >
        {loadingList ? (
          <LoadingSkeleton variant="row" lines={4} />
        ) : receipts.length === 0 ? (
          <EmptyState
            icon={<BarChart3 className="h-7 w-7" />}
            title="No receipts available to compare"
            description="Run at least two evaluations first so statistical comparisons can be computed."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="text-ui-sm w-full text-left">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                  <th className="w-12 px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Include
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Model ID
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Task
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Cost
                  </th>
                  <th
                    className="px-5 py-3 text-right font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Baseline
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                {receipts.map((r) => {
                  const checked = selectedIds.includes(r.dir_name);
                  const isBase = baselineId === r.dir_name;
                  return (
                    <tr
                      key={r.dir_name}
                      style={{
                        backgroundColor: checked ? "var(--bg-elevated)" : "transparent",
                      }}
                    >
                      <td className="px-5 py-3">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleReceipt(r.dir_name)}
                          aria-label={`Include ${r.dir_name}`}
                          className="h-4 w-4 cursor-pointer accent-[var(--accent-cyan)]"
                        />
                      </td>
                      <td className="font-mono-num px-4 py-3 font-medium">{r.model_id}</td>
                      <td className="px-4 py-3" style={{ color: "var(--text-secondary)" }}>
                        {r.task_name}
                      </td>
                      <td className="font-mono-num px-4 py-3">{formatUsd(r.total_cost_usd)}</td>
                      <td className="px-5 py-3 text-right">
                        <label className="text-ui-sm inline-flex cursor-pointer items-center gap-2">
                          <input
                            type="radio"
                            name="baseline-receipt"
                            disabled={!checked}
                            checked={isBase}
                            onChange={() => setBaselineId(r.dir_name)}
                            className="h-4 w-4 accent-[var(--accent-cyan)]"
                          />
                          <span
                            style={{
                              color: isBase ? "var(--accent-cyan)" : "var(--text-secondary)",
                            }}
                          >
                            {isBase ? "Baseline" : "Select"}
                          </span>
                        </label>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
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

          {comparison.pairwise.map((pair) => (
            <Card
              key={`${pair.baseline_id}-${pair.challenger_id}`}
              title={`Head-to-head: ${pair.challenger_id} vs ${pair.baseline_id}`}
              subtitle={`Metric: ${pair.metric}`}
              padding="none"
              actions={
                <StatusBadge status={pair.unresolved ? "UNRESOLVED" : "VERIFIED"} />
              }
            >
              <div
                className="grid grid-cols-1 divide-y border-b sm:grid-cols-4 sm:divide-x sm:divide-y-0"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <div className="p-4">
                  <div className="text-ui-sm" style={{ color: "var(--text-secondary)" }}>
                    Paired bootstrap delta
                  </div>
                  <div className="font-mono-num text-ui-lg mt-1 font-semibold">
                    {(pair.paired_bootstrap.mean_delta * 100).toFixed(2)}%
                  </div>
                  <div className="font-mono-num text-ui-sm mt-0.5" style={{ color: "var(--text-muted)" }}>
                    p = {pair.paired_bootstrap.p_value.toFixed(4)}
                  </div>
                </div>
                <div className="p-4">
                  <div className="text-ui-sm" style={{ color: "var(--text-secondary)" }}>
                    95% bootstrap CI
                  </div>
                  <div className="font-mono-num text-ui-lg mt-1 font-semibold">
                    [{(pair.paired_bootstrap.ci_low * 100).toFixed(1)}%,{" "}
                    {(pair.paired_bootstrap.ci_high * 100).toFixed(1)}%]
                  </div>
                </div>
                <div className="p-4">
                  <div className="text-ui-sm" style={{ color: "var(--text-secondary)" }}>
                    McNemar significance
                  </div>
                  <div className="font-mono-num text-ui-lg mt-1 font-semibold">
                    {pair.mcnemar ? `p = ${pair.mcnemar.p_value.toFixed(4)}` : "N/A"}
                  </div>
                  {pair.mcnemar && (
                    <div className="font-mono-num text-ui-sm mt-0.5" style={{ color: "var(--text-muted)" }}>
                      n01={pair.mcnemar.n_01}, n10={pair.mcnemar.n_10}
                    </div>
                  )}
                </div>
                <div className="p-4">
                  <div className="text-ui-sm" style={{ color: "var(--text-secondary)" }}>
                    Decision flip rate
                  </div>
                  <div className="font-mono-num text-ui-lg mt-1 font-semibold">
                    {pair.noise_floor ? formatMetric(pair.noise_floor.flip_rate) : "--"}
                  </div>
                  {pair.noise_floor && (
                    <div className="font-mono-num text-ui-sm mt-0.5" style={{ color: "var(--text-muted)" }}>
                      {pair.noise_floor.n_flips} / {pair.noise_floor.n_decisions} flips
                    </div>
                  )}
                </div>
              </div>

              {pair.item_diffs.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="text-ui-sm w-full text-left">
                    <thead>
                      <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                        <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          Sample ID
                        </th>
                        <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          Baseline
                        </th>
                        <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          Challenger
                        </th>
                        <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          Target
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                      {pair.item_diffs.slice(0, 20).map((diff) => (
                        <tr key={diff.sample_id}>
                          <td className="font-mono-num px-5 py-2.5">{diff.sample_id}</td>
                          <td className="font-mono-num px-4 py-2.5">
                            {diff.baseline_score} ({diff.baseline_output.slice(0, 36)})
                          </td>
                          <td className="font-mono-num px-4 py-2.5">
                            {diff.challenger_score} ({diff.challenger_output.slice(0, 36)})
                          </td>
                          <td
                            className="font-mono-num px-5 py-2.5"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            {String(diff.target).slice(0, 36)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-ui-sm px-5 py-4" style={{ color: "var(--text-secondary)" }}>
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
