import { RefreshCw, Search, ShieldCheck, ShieldX } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import {
  Badge,
  Card,
  CodeBlock,
  CopyButton,
  EmptyState,
  ErrorBanner,
  LoadingSkeleton,
  MetricPill,
  SlideDrawer,
} from "../components/common";
import type {
  ReceiptDetailResponse,
  ReceiptListItem,
  ScoredSample,
  VerifyReceiptResponse,
} from "../types/evalStudio";
import { formatDateTime, formatUsd, shortHash } from "../utils/format";

export function ReceiptVaultPage() {
  const [receipts, setReceipts] = useState<ReceiptListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ReceiptDetailResponse | null>(null);
  const [samples, setSamples] = useState<ScoredSample[]>([]);
  const [query, setQuery] = useState("");
  const [verifyResult, setVerifyResult] = useState<VerifyReceiptResponse | null>(null);
  const [drawerSample, setDrawerSample] = useState<ScoredSample | null>(null);
  const [loading, setLoading] = useState(true);
  const [rescoring, setRescoring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadList = async () => {
    setLoading(true);
    try {
      const res = await api.listReceipts();
      setReceipts(res.receipts);
      if (res.receipts.length > 0 && !selectedId) {
        setSelectedId(res.receipts[0]!.dir_name);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load receipts.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadList();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    setVerifyResult(null);
    Promise.all([
      api.getReceipt(selectedId, controller.signal),
      api.getReceiptSamples(selectedId, { limit: 100, q: query || undefined }, controller.signal),
    ])
      .then(([d, s]) => {
        setDetail(d);
        setSamples(s.samples);
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name !== "AbortError") setError(err.message);
      });
    return () => controller.abort();
  }, [selectedId, query]);

  const handleVerify = async () => {
    if (!selectedId) return;
    setError(null);
    try {
      const res = await api.verifyReceipt(selectedId);
      setVerifyResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Verification failed.");
    }
  };

  const handleOfflineRescore = async () => {
    if (!selectedId) return;
    setRescoring(true);
    setError(null);
    try {
      const res = await api.rescoreReceipt(selectedId, {});
      await loadList();
      setSelectedId(res.receipt_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Offline re-scoring failed.");
    } finally {
      setRescoring(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Portable Eval Receipts" subtitle="Click any receipt to inspect & verify">
          {loading ? (
            <LoadingSkeleton lines={5} />
          ) : receipts.length === 0 ? (
            <EmptyState
              icon={<ShieldCheck className="h-8 w-8" />}
              title="Vault is empty"
              description="Completed runs will automatically appear here as verifiable receipts."
            />
          ) : (
            <div className="max-h-[34rem] space-y-2 overflow-y-auto">
              {receipts.map((r) => {
                const active = r.dir_name === selectedId;
                return (
                  <button
                    key={r.dir_name}
                    type="button"
                    onClick={() => setSelectedId(r.dir_name)}
                    className="w-full rounded-xl border p-3 text-left transition-colors"
                    style={{
                      borderColor: active ? "var(--accent-cyan)" : "var(--border-subtle)",
                      backgroundColor: active ? "var(--accent-cyan-soft)" : "var(--bg-elevated)",
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold">{r.task_name}</span>
                      <Badge variant="violet">{r.model_id}</Badge>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[11px]" style={{ color: "var(--text-secondary)" }}>
                      <span>{formatDateTime(r.created_at_utc)}</span>
                      <span className="font-mono">{formatUsd(r.total_cost_usd)}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </Card>

        <div className="space-y-6 lg:col-span-2">
          {detail && (
            <>
              <Card
                title={`${detail.manifest.task_name} · ${detail.manifest.model_id}`}
                subtitle={`Run ID: ${detail.manifest.run_id} · SHA: ${shortHash(detail.manifest.manifest_sha256)}`}
                actions={
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleVerify()}
                      className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-semibold"
                      style={{ backgroundColor: "var(--accent-emerald)", color: "#090D16" }}
                    >
                      <ShieldCheck className="h-3.5 w-3.5" /> Verify SHA-256
                    </button>
                    <button
                      type="button"
                      disabled={rescoring}
                      onClick={() => void handleOfflineRescore()}
                      className="inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-xs font-semibold disabled:opacity-50"
                      style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
                    >
                      <RefreshCw className="h-3.5 w-3.5" /> Offline Re-Score
                    </button>
                  </div>
                }
              >
                {verifyResult && (
                  <div
                    className="mb-4 flex items-start gap-2 rounded-xl p-3 text-xs"
                    style={{
                      backgroundColor: verifyResult.valid
                        ? "var(--accent-emerald-soft)"
                        : "var(--accent-rose-soft)",
                      color: verifyResult.valid ? "var(--accent-emerald)" : "var(--accent-rose)",
                    }}
                  >
                    {verifyResult.valid ? (
                      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
                    ) : (
                      <ShieldX className="mt-0.5 h-4 w-4 shrink-0" />
                    )}
                    <div>
                      <div className="font-semibold">
                        {verifyResult.valid
                          ? "Cryptographic Integrity Verified (Tamper-Free & Secret-Free)"
                          : "Integrity Audit Failed"}
                      </div>
                      {verifyResult.errors.map((err, i) => (
                        <div key={i}>{err}</div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mb-4 flex flex-wrap gap-2">
                  {Object.entries(detail.manifest.metrics).map(([k, m]) => (
                    <MetricPill
                      key={k}
                      label={k}
                      value={m.value}
                      stderr={m.stderr}
                      unresolved={m.unresolved}
                    />
                  ))}
                </div>

                {detail.manifest.reproduce_command && (
                  <div
                    className="flex items-center justify-between gap-2 rounded-lg border px-3 py-2 font-mono text-[11px]"
                    style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-canvas)" }}
                  >
                    <span className="truncate">{detail.manifest.reproduce_command}</span>
                    <CopyButton value={detail.manifest.reproduce_command} label="Copy CLI" />
                  </div>
                )}
              </Card>

              <Card
                title={`Sample Inspector (${samples.length} / ${detail.total_samples})`}
                subtitle="Click any row to inspect raw outputs, filtered answers, and reasoning traces"
                actions={
                  <div className="relative">
                    <Search
                      className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5"
                      style={{ color: "var(--text-muted)" }}
                    />
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Filter samples..."
                      className="rounded-lg border py-1.5 pl-8 pr-3 text-xs"
                      style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
                    />
                  </div>
                }
              >
                <div className="max-h-96 overflow-y-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                        <th className="py-2 pr-3">ID</th>
                        <th className="py-2 pr-3">Prompt</th>
                        <th className="py-2 pr-3">Target</th>
                        <th className="py-2 text-right">Response</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samples.map((s) => (
                        <tr
                          key={s.sample_id}
                          onClick={() => setDrawerSample(s)}
                          className="cursor-pointer border-b transition-colors hover:opacity-85 last:border-b-0"
                          style={{ borderColor: "var(--border-subtle)" }}
                        >
                          <td className="py-2 pr-3 font-mono text-[11px]">{s.sample_id}</td>
                          <td className="max-w-xs truncate py-2 pr-3">{s.input_text}</td>
                          <td className="py-2 pr-3 font-mono text-[11px]">{String(s.target)}</td>
                          <td className="max-w-xs truncate py-2 text-right font-mono text-[11px]" style={{ color: "var(--accent-cyan)" }}>
                            {s.raw_responses[0]?.text ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </div>
      </div>

      <SlideDrawer
        open={drawerSample !== null}
        onClose={() => setDrawerSample(null)}
        title={`Sample ${drawerSample?.sample_id ?? ""}`}
        subtitle="Raw response, reasoning trace, and per-filter scores"
      >
        {drawerSample && (
          <div className="space-y-4 text-xs">
            <CodeBlock code={drawerSample.input_text || "(empty)"} title="Input Prompt" />
            <CodeBlock code={String(drawerSample.target)} title="Gold Target" />
            {drawerSample.raw_responses.map((resp, idx) => (
              <div key={idx} className="space-y-2 rounded-xl border p-3" style={{ borderColor: "var(--border-subtle)" }}>
                <div className="flex items-center justify-between">
                  <span className="font-semibold">Response #{idx + 1}</span>
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    {resp.model_id} · {resp.latency_ms.toFixed(0)} ms
                  </span>
                </div>
                {resp.reasoning_trace && (
                  <CodeBlock code={resp.reasoning_trace} title="Reasoning Trace (<think>)" />
                )}
                <CodeBlock code={resp.text ?? "(null)"} title="Final Answer" />
              </div>
            ))}
            <CodeBlock
              code={JSON.stringify(drawerSample.metrics, null, 2)}
              language="json"
              title="Per-Filter Metric Scores"
            />
          </div>
        )}
      </SlideDrawer>
    </div>
  );
}
