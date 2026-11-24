import { Calculator, Play, ShieldAlert, Terminal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { Badge, Card, ErrorBanner } from "../components/common";
import type {
  CostEstimateResponse,
  ProviderPreset,
  RunSseEvent,
  TaskSummary,
} from "../types/evalStudio";
import { formatTokens, formatUsd } from "../utils/format";

export function RunLauncherPage() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [presets, setPresets] = useState<ProviderPreset[]>([]);
  const [taskPath, setTaskPath] = useState("");
  const [backend, setBackend] = useState<"mock" | "http-byok">("mock");
  const [modelId, setModelId] = useState("mock-deterministic");
  const [baseUrl, setBaseUrl] = useState("");
  const [judgeModel, setJudgeModel] = useState("");
  const [fallbackJudgeModel, setFallbackJudgeModel] = useState("");
  const [limit, setLimit] = useState<string>("");
  const [repeats, setRepeats] = useState(1);
  const [bootstrapIters, setBootstrapIters] = useState(1000);
  const [maxCostUsd, setMaxCostUsd] = useState<string>("1.00");
  const [olmesEnabled, setOlmesEnabled] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);
  const [thinkEndToken, setThinkEndToken] = useState("</think>");

  const [estimate, setEstimate] = useState<CostEstimateResponse | null>(null);
  const [events, setEvents] = useState<RunSseEvent[]>([]);
  const [runStatus, setRunStatus] = useState<string>("idle");
  const [spentUsd, setSpentUsd] = useState(0);
  const [completedSamples, setCompletedSamples] = useState(0);
  const [totalSamples, setTotalSamples] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.listTasks(controller.signal),
      api.getByokPresets(controller.signal),
    ])
      .then(([t, p]) => {
        setTasks(t.tasks);
        setPresets(p.presets);
        if (t.tasks.length > 0 && !taskPath) {
          setTaskPath(t.tasks[0]!.path);
        }
      })
      .catch((err: unknown) => {
        if (err instanceof Error && err.name !== "AbortError") setError(err.message);
      });
    return () => {
      controller.abort();
      cleanupRef.current?.();
    };
  }, []);

  const buildPayload = () => ({
    task_path: taskPath,
    model_backend: backend,
    model_id: modelId,
    base_url: baseUrl.trim() || undefined,
    judge_model: judgeModel.trim() || undefined,
    fallback_judge_model: fallbackJudgeModel.trim() || undefined,
    limit: limit.trim() ? Number(limit) : null,
    repeats,
    olmes_enabled: olmesEnabled,
    enable_thinking: enableThinking,
    think_end_token: enableThinking ? thinkEndToken : null,
    bootstrap_iters: bootstrapIters,
    max_cost_usd: maxCostUsd.trim() ? Number(maxCostUsd) : null,
  });

  const handleEstimate = async () => {
    if (!taskPath) return;
    setError(null);
    try {
      const res = await api.estimateRun(buildPayload());
      setEstimate(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Cost estimation failed.");
    }
  };

  const handleLaunch = async () => {
    if (!taskPath) return;
    cleanupRef.current?.();
    setError(null);
    setEvents([]);
    setSpentUsd(0);
    setCompletedSamples(0);
    setTotalSamples(0);
    setRunStatus("launching");
    try {
      const launched = await api.launchRun(buildPayload());
      setRunStatus("running");
      cleanupRef.current = api.subscribeRunEvents(launched.run_id, {
        onEvent: (ev) => setEvents((prev) => [...prev, ev]),
        onRunStarted: (ev) => setTotalSamples(ev.total_samples),
        onSampleScored: (ev) => {
          setCompletedSamples((c) => Math.max(c + 1, ev.index + 1));
          setTotalSamples(ev.total);
          setSpentUsd(ev.spent_usd);
        },
        onBudgetExceeded: (ev) => {
          setSpentUsd(ev.spent_usd);
          setRunStatus("budget_exceeded");
          setError(ev.error);
        },
        onCompleted: (ev) => {
          setSpentUsd(ev.spent_usd);
          setRunStatus("completed");
        },
        onError: (err) => {
          setRunStatus("failed");
          setError(err.message);
        },
      });
    } catch (err: unknown) {
      setRunStatus("failed");
      setError(err instanceof Error ? err.message : "Failed to launch run.");
    }
  };

  const applyPreset = (presetId: string) => {
    const preset = presets.find((p) => p.id === presetId);
    if (!preset) return;
    setBackend(preset.id === "mock" ? "mock" : "http-byok");
    setBaseUrl(preset.default_base_url);
    if (preset.models[0]) {
      setModelId(preset.models[0].id);
      setEnableThinking(preset.models[0].supports_thinking);
    }
  };

  const pct = totalSamples > 0 ? Math.min(100, Math.round((completedSamples / totalSamples) * 100)) : 0;

  return (
    <div className="space-y-6">
      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Run Configuration Builder */}
        <Card
          title="Evaluation Run Configuration"
          subtitle="Select benchmark, model preset, judge, and safety caps"
          className="lg:col-span-2"
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Benchmark Task YAML</span>
              <select
                value={taskPath}
                onChange={(e) => setTaskPath(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              >
                {tasks.map((t) => (
                  <option key={t.path} value={t.path}>
                    {t.name} ({t.source})
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Quick Provider Preset</span>
              <select
                defaultValue=""
                onChange={(e) => applyPreset(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              >
                <option value="">Choose preset (DeepSeek, Sarvam, OpenAI...)</option>
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} — {p.models[0]?.id ?? ""}
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Backend</span>
              <select
                value={backend}
                onChange={(e) => setBackend(e.target.value as "mock" | "http-byok")}
                className="w-full rounded-lg border px-3 py-2 text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              >
                <option value="mock">mock (offline deterministic)</option>
                <option value="http-byok">http-byok (DeepSeek / Sarvam / OpenAI / Ollama)</option>
              </select>
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Model ID</span>
              <input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="deepseek-flash or glm5.3-flash"
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              />
            </label>

            <label className="block space-y-1 text-xs sm:col-span-2">
              <span style={{ color: "var(--text-secondary)" }}>Custom Base URL (optional)</span>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.deepseek.com or https://api.sarvam.ai/v2"
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              />
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Judge Model (optional)</span>
              <input
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
                placeholder="mock or gpt-4o-mini"
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              />
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Fallback Judge (optional)</span>
              <input
                value={fallbackJudgeModel}
                onChange={(e) => setFallbackJudgeModel(e.target.value)}
                placeholder="mock"
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              />
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Hard Budget Cap (--max-cost-usd)</span>
              <input
                type="number"
                step="0.05"
                min="0"
                value={maxCostUsd}
                onChange={(e) => setMaxCostUsd(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              />
            </label>

            <div className="grid grid-cols-3 gap-2">
              <label className="block space-y-1 text-xs">
                <span style={{ color: "var(--text-secondary)" }}>Limit</span>
                <input
                  type="number"
                  min="1"
                  value={limit}
                  onChange={(e) => setLimit(e.target.value)}
                  placeholder="All"
                  className="w-full rounded-lg border px-2 py-2 font-mono text-xs"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-elevated)",
                  }}
                />
              </label>
              <label className="block space-y-1 text-xs">
                <span style={{ color: "var(--text-secondary)" }}>Repeats</span>
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={repeats}
                  onChange={(e) => setRepeats(Number(e.target.value) || 1)}
                  className="w-full rounded-lg border px-2 py-2 font-mono text-xs"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-elevated)",
                  }}
                />
              </label>
              <label className="block space-y-1 text-xs">
                <span style={{ color: "var(--text-secondary)" }}>Bootstrap</span>
                <input
                  type="number"
                  min="0"
                  step="100"
                  value={bootstrapIters}
                  onChange={(e) => setBootstrapIters(Number(e.target.value) || 0)}
                  className="w-full rounded-lg border px-2 py-2 font-mono text-xs"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-elevated)",
                  }}
                />
              </label>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-5 text-xs">
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={olmesEnabled}
                onChange={(e) => setOlmesEnabled(e.target.checked)}
              />
              <span>OLMES Dual Cloze + MCF</span>
            </label>
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={enableThinking}
                onChange={(e) => setEnableThinking(e.target.checked)}
              />
              <span>Enable Thinking Trace Extraction</span>
            </label>
            {enableThinking && (
              <input
                value={thinkEndToken}
                onChange={(e) => setThinkEndToken(e.target.value)}
                className="w-32 rounded-md border px-2 py-1 font-mono text-xs"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-elevated)",
                }}
              />
            )}
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void handleEstimate()}
              className="inline-flex items-center gap-1.5 rounded-lg border px-4 py-2 text-xs font-semibold"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
            >
              <Calculator className="h-3.5 w-3.5" /> Estimate Cost
            </button>
            <button
              type="button"
              onClick={() => void handleLaunch()}
              disabled={runStatus === "running"}
              className="inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-xs font-semibold disabled:opacity-50"
              style={{ backgroundColor: "var(--accent-cyan)", color: "#090D16" }}
            >
              <Play className="h-3.5 w-3.5" /> Launch Run
            </button>
          </div>
        </Card>

        {/* Pre-Flight Cost Breakdown */}
        <Card title="Pre-Flight Cost Breakdown" subtitle="Target + Judge token estimate" elevated>
          {estimate ? (
            <div className="space-y-3 text-xs">
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>Samples × Repeats</span>
                <span className="font-mono font-semibold">
                  {estimate.n_samples} × {estimate.repeats}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>Target Tokens (in / out)</span>
                <span className="font-mono">
                  {formatTokens(estimate.target_prompt_tokens)} /{" "}
                  {formatTokens(estimate.target_completion_tokens)}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>Target Est. USD</span>
                <span className="font-mono">{formatUsd(estimate.target_usd)}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>Judge Est. USD</span>
                <span className="font-mono">{formatUsd(estimate.judge_usd)}</span>
              </div>
              <div
                className="flex items-center justify-between border-t pt-3 text-sm font-semibold"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <span>Total Estimate</span>
                <span className="font-mono" style={{ color: "var(--accent-cyan)" }}>
                  {formatUsd(estimate.total_usd)}
                </span>
              </div>
              {estimate.exceeds_max_cost && (
                <div
                  className="flex items-center gap-2 rounded-lg p-2.5 text-[11px]"
                  style={{ backgroundColor: "var(--accent-rose-soft)", color: "var(--accent-rose)" }}
                >
                  <ShieldAlert className="h-4 w-4 shrink-0" />
                  Exceeds --max-cost-usd cap ({formatUsd(Number(maxCostUsd))})!
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
              Click <strong>Estimate Cost</strong> to compute exact token & USD projections before
              dispatching requests.
            </div>
          )}
        </Card>
      </div>

      {/* Live SSE Streaming Telemetry Console */}
      <Card
        title="Live SSE Telemetry Console"
        subtitle="Real-time sample scoring, reasoning trace snippets, and spend meter"
        actions={
          <div className="flex items-center gap-2">
            <Badge
              variant={
                runStatus === "completed"
                  ? "emerald"
                  : runStatus === "running"
                    ? "cyan"
                    : runStatus === "failed" || runStatus === "budget_exceeded"
                      ? "rose"
                      : "neutral"
              }
              dot
            >
              {runStatus.toUpperCase()}
            </Badge>
            <span className="font-mono text-xs tabular-nums">
              {completedSamples}/{totalSamples} ({pct}%) · {formatUsd(spentUsd)}
            </span>
          </div>
        }
      >
        <div className="mb-3 h-2 w-full overflow-hidden rounded-full" style={{ backgroundColor: "var(--bg-elevated)" }}>
          <div
            className="h-full transition-all duration-300"
            style={{ width: `${pct}%`, backgroundColor: "var(--accent-cyan)" }}
          />
        </div>
        <div
          className="max-h-72 overflow-y-auto rounded-lg border p-3 font-mono text-[11px]"
          style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-canvas)" }}
        >
          {events.length === 0 ? (
            <div className="flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
              <Terminal className="h-3.5 w-3.5" /> Waiting for SSE stream events...
            </div>
          ) : (
            events.map((ev, i) => (
              <div key={i} className="border-b py-1 last:border-b-0" style={{ borderColor: "var(--border-subtle)" }}>
                <span style={{ color: "var(--accent-cyan)" }}>[{ev.event}]</span>{" "}
                <span style={{ color: "var(--text-secondary)" }}>{JSON.stringify(ev).slice(0, 220)}</span>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
