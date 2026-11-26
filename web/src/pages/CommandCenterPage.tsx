import { Activity, ArrowRight, DollarSign, FileCheck2, KeyRound, Play, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import {
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
      name: `${r.model_id.slice(0, 14)} (${r.task_name.slice(0, 10)})`,
      score: firstMetric ? Number((firstMetric.value * 100).toFixed(1)) : 0,
    };
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1
            className="text-[26px] font-semibold tracking-[-0.03em]"
            style={{ color: "var(--text-primary)" }}
          >
            Evaluation command center
          </h1>
          <p className="text-ui-base mt-1 max-w-[62ch]" style={{ color: "var(--text-secondary)" }}>
            Monitor verifiable receipts, benchmark packs, API credentials, and model accuracy from a single local workspace.
          </p>
        </div>
        <Link
          to="/run"
          className="telemetry-transition text-ui-base inline-flex items-center gap-2 rounded-md px-4 py-2 font-semibold hover:opacity-90"
          style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
        >
          <Play className="h-4 w-4" /> Launch evaluation
        </Link>
      </div>

      {error && <ErrorBanner error={error} />}

      <div className="structural-grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          label="Portable receipts"
          value={receipts.length}
          hint="SHA-256 verifiable runs"
          accent="cyan"
          icon={<FileCheck2 className="h-4 w-4" />}
          loading={loading}
        />
        <KpiTile
          label="Registered benchmarks"
          value={tasks.length}
          hint="Built-in, BYOT, and BYOP packs"
          accent="cyan"
          icon={<Sparkles className="h-4 w-4" />}
          loading={loading}
        />
        <KpiTile
          label="Cumulative spend"
          value={formatUsd(totalSpend)}
          hint="Tracked across local receipts"
          accent="emerald"
          icon={<DollarSign className="h-4 w-4" />}
          loading={loading}
        />
        <KpiTile
          label="Active BYOK roles"
          value={`${configuredRoles} / ${byok?.roles.length ?? 4}`}
          hint="Session, env, and OS keyring"
          accent="amber"
          icon={<KeyRound className="h-4 w-4" />}
          loading={loading}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <Card
          title="Recent model performance"
          subtitle="Primary metric score (%) across recent portable receipts"
          className="lg:col-span-8"
        >
          {loading ? (
            <LoadingSkeleton variant="chart" lines={5} />
          ) : overviewChartData.length === 0 ? (
            <EmptyState
              icon={<Activity className="h-7 w-7" />}
              title="No evaluation receipts recorded"
              description="Launch an offline mock or live BYOK run to populate benchmark telemetry."
              action={
                <Link
                  to="/run"
                  className="telemetry-transition text-ui-base inline-flex items-center gap-2 rounded-md px-4 py-2 font-semibold hover:opacity-90"
                  style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
                >
                  <Play className="h-4 w-4" /> Launch first run
                </Link>
              }
            />
          ) : (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={overviewChartData} margin={{ top: 10, right: 16, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                  <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 13 }} />
                  <YAxis
                    domain={[0, 100]}
                    unit="%"
                    tick={{ fill: "var(--text-secondary)", fontSize: 13 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--bg-elevated)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "6px",
                      fontSize: "13px",
                      color: "var(--text-primary)",
                    }}
                  />
                  <Bar dataKey="score" name="Score (%)" fill="#06B6D4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card
          title="Workflow directory"
          subtitle="Jump directly to any evaluation stage"
          padding="none"
          className="lg:col-span-4"
        >
          <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
            <Link
              to="/run"
              className="telemetry-transition flex items-center justify-between gap-3 px-5 py-4 hover:bg-[var(--bg-elevated)]"
            >
              <div>
                <div className="text-ui-base font-semibold">Launch evaluation run</div>
                <div className="text-ui-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
                  Pre-flight cost estimate, USD cap, and live SSE stream
                </div>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0" style={{ color: "var(--accent-cyan)" }} />
            </Link>

            <Link
              to="/papers"
              className="telemetry-transition flex items-center justify-between gap-3 px-5 py-4 hover:bg-[var(--bg-elevated)]"
            >
              <div>
                <div className="text-ui-base font-semibold">Compile scientific papers</div>
                <div className="text-ui-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
                  Two-stage grounded extraction and SHA-256 suite locking
                </div>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0" style={{ color: "var(--text-muted)" }} />
            </Link>

            <Link
              to="/tests"
              className="telemetry-transition flex items-center justify-between gap-3 px-5 py-4 hover:bg-[var(--bg-elevated)]"
            >
              <div>
                <div className="text-ui-base font-semibold">Author and calibrate tests</div>
                <div className="text-ui-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
                  Sandboxed Jinja2 preview and Cohen&apos;s kappa calibration
                </div>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0" style={{ color: "var(--text-muted)" }} />
            </Link>

            <Link
              to="/compare"
              className="telemetry-transition flex items-center justify-between gap-3 px-5 py-4 hover:bg-[var(--bg-elevated)]"
            >
              <div>
                <div className="text-ui-base font-semibold">Compare model receipts</div>
                <div className="text-ui-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
                  Confidence intervals, McNemar test, and Pareto scatter
                </div>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0" style={{ color: "var(--text-muted)" }} />
            </Link>
          </div>
        </Card>
      </div>

      <Card
        title="Recent portable receipts"
        subtitle="Verified receipts discovered in the local workspace"
        padding="none"
        actions={
          <Link
            to="/receipts"
            className="text-ui-sm font-medium"
            style={{ color: "var(--accent-cyan)" }}
          >
            Open receipt vault →
          </Link>
        }
      >
        {loading ? (
          <LoadingSkeleton variant="row" lines={4} />
        ) : receipts.length === 0 ? (
          <div className="text-ui-sm px-5 py-6" style={{ color: "var(--text-secondary)" }}>
            No receipts found yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-ui-sm w-full text-left">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                  <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Task
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Model
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Metrics
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Cost
                  </th>
                  <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                    Manifest SHA-256
                  </th>
                  <th
                    className="px-5 py-3 text-right font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Timestamp
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                {receipts.slice(0, 10).map((r) => (
                  <tr key={r.dir_name} className="hover:bg-[var(--bg-elevated)]">
                    <td className="px-5 py-3 font-medium">{r.task_name}</td>
                    <td
                      className="font-mono-num px-4 py-3"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {r.model_id}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-x-4 gap-y-1">
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
                    <td className="font-mono-num px-4 py-3">{formatUsd(r.total_cost_usd)}</td>
                    <td
                      className="font-mono-num px-4 py-3"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {shortHash(r.manifest_sha256)}
                    </td>
                    <td
                      className="px-5 py-3 text-right"
                      style={{ color: "var(--text-secondary)" }}
                    >
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
