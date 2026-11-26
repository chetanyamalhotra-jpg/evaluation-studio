import { RefreshCw, Search, ShieldCheck, ShieldX } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import {
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
      <div>
        <h1
          className="text-[26px] font-semibold tracking-[-0.03em]"
          style={{ color: "var(--text-primary)" }}
        >
          Portable receipt vault and sample traces
        </h1>
        <p className="text-ui-base mt-1 max-w-[64ch]" style={{ color: "var(--text-secondary)" }}>
          Audit cryptographic SHA-256 hashes, re-score runs offline, copy CLI reproduce commands, and inspect sample reasoning traces.
        </p>
      </div>

      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <Card
          title="Recorded receipts"
          subtitle="Select a run directory to inspect"
          padding="none"
          className="lg:col-span-4"
        >
          {loading ? (
            <LoadingSkeleton variant="row" lines={5} />
          ) : receipts.length === 0 ? (
            <EmptyState
              icon={<ShieldCheck className="h-7 w-7" />}
              title="Receipt vault is empty"
              description="Completed runs appear here automatically as verifiable SHA-256 bundles."
            />
          ) : (
            <div
              className="max-h-[36rem] divide-y overflow-y-auto"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              {receipts.map((r) => {
                const active = r.dir_name === selectedId;
                return (
                  <button
                    key={r.dir_name}
                    type="button"
                    onClick={() => setSelectedId(r.dir_name)}
                    className="telemetry-transition relative block w-full cursor-pointer px-5 py-3.5 text-left hover:bg-[var(--bg-elevated)]"
                    style={{
                      backgroundColor: active ? "var(--bg-elevated)" : "transparent",
                    }}
                  >
                    {active && (
                      <span
                        className="absolute inset-y-0 left-0 w-[2px]"
                        style={{ backgroundColor: "var(--accent-cyan)" }}
                      />
                    )}
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-ui-base truncate font-semibold">{r.task_name}</span>
                      <span
                        className="font-mono-num text-ui-sm shrink-0"
                        style={{ color: "var(--accent-cyan)" }}
                      >
                        {r.model_id}
                      </span>
                    </div>
                    <div
                      className="text-ui-sm mt-1 flex items-center justify-between"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <span>{formatDateTime(r.created_at_utc)}</span>
                      <span className="font-mono-num">{formatUsd(r.total_cost_usd)}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </Card>

        <div className="space-y-6 lg:col-span-8">
          {detail && (
            <>
              <Card
                title={`${detail.manifest.task_name} (${detail.manifest.model_id})`}
                subtitle={`Run ID: ${detail.manifest.run_id} | SHA-256: ${shortHash(detail.manifest.manifest_sha256)}`}
                padding="none"
                actions={
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      disabled={rescoring}
                      onClick={() => void handleOfflineRescore()}
                      className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md border px-3 py-1.5 font-medium hover:bg-[var(--bg-elevated)] disabled:opacity-50"
                      style={{ borderColor: "var(--border-subtle)", color: "var(--text-primary)" }}
                    >
                      <RefreshCw className="h-4 w-4" /> Offline re-score
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleVerify()}
                      className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md px-3.5 py-1.5 font-semibold hover:opacity-90"
                      style={{
                        backgroundColor: "var(--accent-emerald)",
                        color: "var(--text-inverse)",
                      }}
                    >
                      <ShieldCheck className="h-4 w-4" /> Verify SHA-256
                    </button>
                  </div>
                }
              >
                <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                  {verifyResult && (
                    <div
                      className="text-ui-sm flex items-start gap-2.5 px-5 py-3.5"
                      style={{
                        backgroundColor: verifyResult.valid
                          ? "var(--accent-emerald-soft)"
                          : "var(--accent-rose-soft)",
                        color: verifyResult.valid
                          ? "var(--accent-emerald)"
                          : "var(--accent-rose)",
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
                            ? "Cryptographic integrity verified (tamper-free and secret-free)"
                            : "Integrity audit failed"}
                        </div>
                        {verifyResult.errors.map((err, i) => (
                          <div key={i} className="mt-0.5">
                            {err}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-x-6 gap-y-2 px-5 py-4">
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
                      className="flex items-center justify-between gap-3 px-5 py-3"
                      style={{ backgroundColor: "var(--bg-code)" }}
                    >
                      <span className="font-mono-num text-ui-sm truncate">
                        {detail.manifest.reproduce_command}
                      </span>
                      <CopyButton value={detail.manifest.reproduce_command} label="Copy CLI" />
                    </div>
                  )}
                </div>
              </Card>

              <Card
                title={`Sample inspector (${samples.length} / ${detail.total_samples})`}
                subtitle="Select any row to inspect raw responses, reasoning traces, and scores"
                padding="none"
                actions={
                  <div className="relative">
                    <Search
                      className="pointer-events-none absolute left-3 top-2.5 h-4 w-4"
                      style={{ color: "var(--text-muted)" }}
                    />
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Filter samples..."
                      className="input-control py-1.5 pl-9 pr-3"
                    />
                  </div>
                }
              >
                <div className="max-h-96 overflow-y-auto">
                  <table className="text-ui-sm w-full text-left">
                    <thead>
                      <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                        <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          ID
                        </th>
                        <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          Prompt
                        </th>
                        <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                          Target
                        </th>
                        <th
                          className="px-5 py-3 text-right font-medium"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Response
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                      {samples.map((s) => (
                        <tr
                          key={s.sample_id}
                          onClick={() => setDrawerSample(s)}
                          className="cursor-pointer hover:bg-[var(--bg-elevated)]"
                        >
                          <td className="font-mono-num px-5 py-3">{s.sample_id}</td>
                          <td className="max-w-xs truncate px-4 py-3">{s.input_text}</td>
                          <td className="font-mono-num px-4 py-3">{String(s.target)}</td>
                          <td
                            className="font-mono-num max-w-xs truncate px-5 py-3 text-right"
                            style={{ color: "var(--accent-cyan)" }}
                          >
                            {s.raw_responses[0]?.text ?? "--"}
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
          <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
            <CodeBlock
              code={drawerSample.input_text || "(empty)"}
              title="Input prompt"
              flush
            />
            <CodeBlock
              code={String(drawerSample.target)}
              title="Reference target"
              flush
            />
            {drawerSample.raw_responses.map((resp, idx) => (
              <div key={idx} className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                <div
                  className="text-ui-sm flex items-center justify-between px-4 py-2.5"
                  style={{ backgroundColor: "var(--bg-elevated)" }}
                >
                  <span className="font-semibold">Response #{idx + 1}</span>
                  <span className="font-mono-num" style={{ color: "var(--text-secondary)" }}>
                    {resp.model_id} · {resp.latency_ms.toFixed(0)} ms
                  </span>
                </div>
                {resp.reasoning_trace && (
                  <CodeBlock
                    code={resp.reasoning_trace}
                    title="Reasoning trace"
                    flush
                  />
                )}
                <CodeBlock
                  code={resp.text ?? "(null)"}
                  title="Model output"
                  flush
                />
              </div>
            ))}
            <CodeBlock
              code={JSON.stringify(drawerSample.metrics, null, 2)}
              language="json"
              title="Per-filter metric scores"
              flush
            />
          </div>
        )}
      </SlideDrawer>
    </div>
  );
}
