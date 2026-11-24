import { Activity, ArrowRight, DollarSign, FileCheck2, KeyRound, Play, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import {
  Badge,
  Card,
  EmptyState,
  ErrorBanner,
  KpiTile,
  LoadingSkeleton,
  MetricPill,
} from "../components/common";
import type { ByokStatusResponse, ReceiptListItem, TaskSummary } from "../types/evalStudio";
import { formatDateTime, formatUsd, shortHash } from "../utils/format";

export function CommandCenterPage() {
  const [receipts, setReceipts] = useState<ReceiptListItem[]>([]);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [byok, setByok] = useState<ByokStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.listReceipts(controller.signal),
      api.listTasks(controller.signal),
      api.getByokStatus(controller.signal),
    ])
      .then(([r, t, b]) => {
        setReceipts(r.receipts);
        setTasks(t.tasks);
        setByok(b);
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name !== "AbortError") {
          setError(err.message);
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  const totalSpend = receipts.reduce((acc, r) => acc + r.total_cost_usd, 0);
  const configuredRoles = byok ? byok.roles.filter((r) => r.configured).length : 0;

  const overviewChartData = receipts.slice(0, 8).map((r) => {
    const firstMetric = Object.values(r.metrics)[0];
    return {
      name: `${r.model_id.slice(0, 14)} · ${r.task_name.slice(0, 12)}`,
      score: firstMetric ? Number((firstMetric.value * 100).toFixed(1)) : 0,
    };
  });

  return (
    <div className="space-y-6">
      {error && <ErrorBanner error={error} />}

      {/* KPI Ribbon */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          label="Portable Receipts"
          value={receipts.length}
          hint="SHA-256 verifiable runs"
          accent="cyan"
          icon={<FileCheck2 className="h-4 w-4" />}
          loading={loading}
        />
        <KpiTile
          label="Active Benchmarks"
          value={tasks.length}
          hint="Built-in, BYOT & BYOP packs"
          accent="violet"
          icon={<Sparkles className="h-4 w-4" />}
          loading={loading}
        />
        <KpiTile
          label="Cumulative Spend"
          value={formatUsd(totalSpend)}
          hint="Tracked across local receipts"
          accent="emerald"
          icon={<DollarSign className="h-4 w-4" />}
          loading={loading}
        />
        <KpiTile
          label="BYOK Roles Active"
          value={`${configuredRoles} / ${byok?.roles.length ?? 4}`}
          hint="Session, env & OS keyring"
          accent="amber"
          icon={<KeyRound className="h-4 w-4" />}
          loading={loading}
        />
      </div>

      {/* Quick Actions & Multi-Model Chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card
          title="Quick Launch Workbench"
          subtitle="Jump straight into your evaluation workflow"
          elevated
        >
          <div className="space-y-3">
            <Link
              to="/run"
              className="flex items-center justify-between rounded-xl border p-3.5 transition-all hover:translate-x-0.5"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--accent-cyan-soft)",
              }}
            >
              <div className="flex items-center gap-3">
                <Play className="h-4 w-4" style={{ color: "var(--accent-cyan)" }} />
                <div>
                  <div className="text-sm font-semibold">Launch Evaluation Run</div>
                  <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    Cost estimate, USD cap & live SSE telemetry
                  </div>
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </Link>

            <Link
              to="/papers"
              className="flex items-center justify-between rounded-xl border p-3.5 transition-all hover:translate-x-0.5"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-surface)" }}
            >
              <div>
                <div className="text-sm font-semibold">Compile a Paper (BYOP)</div>
                <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  2-stage grounded extraction + SHA-256 lock
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </Link>

            <Link
              to="/compare"
              className="flex items-center justify-between rounded-xl border p-3.5 transition-all hover:translate-x-0.5"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-surface)" }}
            >
              <div>
                <div className="text-sm font-semibold">Compare Models & Benchmarks</div>
                <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  95% CIs, McNemar, paired bootstrap & Pareto
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </Card>

        <Card
          title="Recent Model × Benchmark Performance"
          subtitle="Primary metric score (%) across recent Portable Eval Receipts"
          className="lg:col-span-2"
        >
          {loading ? (
            <LoadingSkeleton lines={5} />
          ) : overviewChartData.length === 0 ? (
            <EmptyState
              icon={<Activity className="h-8 w-8" />}
              title="No evaluation receipts yet"
              description="Launch a deterministic mock or live BYOK run to populate telemetry charts."
              action={
                <Link
                  to="/run"
                  className="inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-semibold"
                  style={{ backgroundColor: "var(--accent-cyan)", color: "#090D16" }}
                >
                  <Play className="h-3.5 w-3.5" /> Launch First Run
                </Link>
              }
            />
          ) : (
            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={overviewChartData} margin={{ top: 10, right: 16, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                  <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 11 }} />
                  <YAxis domain={[0, 100]} unit="%" tick={{ fill: "var(--text-secondary)", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--bg-elevated)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "10px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="score" name="Score (%)" fill="#06B6D4" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>

      {/* Recent Runs Feed */}
      <Card
        title="Recent Evaluation Runs"
        subtitle="Latest Portable Eval Receipts discovered in receipts/"
        actions={
          <Link to="/receipts" className="text-xs font-medium" style={{ color: "var(--accent-cyan)" }}>
            Open Receipt Vault →
          </Link>
        }
      >
        {loading ? (
          <LoadingSkeleton lines={4} />
        ) : receipts.length === 0 ? (
          <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
            No receipts found yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                  <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                    Task
                  </th>
                  <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                    Model
                  </th>
                  <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                    Metrics
                  </th>
                  <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                    Cost
                  </th>
                  <th className="py-2 pr-4 font-semibold" style={{ color: "var(--text-secondary)" }}>
                    Manifest SHA
                  </th>
                  <th className="py-2 text-right font-semibold" style={{ color: "var(--text-secondary)" }}>
                    Timestamp
                  </th>
                </tr>
              </thead>
              <tbody>
                {receipts.slice(0, 10).map((r) => (
                  <tr
                    key={r.dir_name}
                    className="border-b last:border-b-0"
                    style={{ borderColor: "var(--border-subtle)" }}
                  >
                    <td className="py-2.5 pr-4 font-medium">{r.task_name}</td>
                    <td className="py-2.5 pr-4">
                      <Badge variant="violet">{r.model_id}</Badge>
                    </td>
                    <td className="py-2.5 pr-4">
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(r.metrics)
                          .slice(0, 2)
                          .map(([k, m]) => (
                            <MetricPill
                              key={k}
                              label={k}
                              value={m.value}
                              stderr={m.stderr}
                              unresolved={m.unresolved}
                            />
                          ))}
                      </div>
                    </td>
                    <td className="py-2.5 pr-4 font-mono tabular-nums">{formatUsd(r.total_cost_usd)}</td>
                    <td className="py-2.5 pr-4 font-mono text-[11px]" style={{ color: "var(--text-muted)" }}>
                      {shortHash(r.manifest_sha256)}
                    </td>
                    <td className="py-2.5 text-right" style={{ color: "var(--text-secondary)" }}>
                      {formatDateTime(r.created_at_utc)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
