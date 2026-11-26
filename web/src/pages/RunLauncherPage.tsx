import { Calculator, Play, ShieldAlert, Terminal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { Card, ErrorBanner, StatusBadge } from "../components/common";
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

  const pct =
    totalSamples > 0 ? Math.min(100, Math.round((completedSamples / totalSamples) * 100)) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1
          className="text-[26px] font-semibold tracking-[-0.03em]"
          style={{ color: "var(--text-primary)" }}
        >
          Launch evaluation and live telemetry
        </h1>
        <p className="text-ui-base mt-1 max-w-[64ch]" style={{ color: "var(--text-secondary)" }}>
          Configure target and judge endpoints, inspect pre-flight cost projections, and stream sample execution over SSE.
        </p>
      </div>

      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <Card
          title="Run parameters"
          subtitle="Select benchmark specification, inference gateway, and safety limits"
          className="lg:col-span-8"
        >
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <span className="text-ui-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Execution backend
            </span>
            <div
              role="radiogroup"
              aria-label="Execution backend"
              className="inline-flex divide-x overflow-hidden rounded-md border"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              {(["mock", "http-byok"] as const).map((opt) => {
                const active = backend === opt;
                return (
                  <button
                    key={opt}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    onClick={() => setBackend(opt)}
                    className="telemetry-transition text-ui-sm cursor-pointer px-3.5 py-1.5 font-medium"
                    style={{
                      backgroundColor: active ? "var(--accent-cyan-soft)" : "var(--bg-elevated)",
                      color: active ? "var(--accent-cyan)" : "var(--text-secondary)",
                    }}
                  >
                    {opt === "mock" ? "Mock offline" : "Live HTTP BYOK"}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Benchmark task YAML</span>
              <select
                value={taskPath}
                onChange={(e) => setTaskPath(e.target.value)}
                className="input-control w-full px-3 py-2"
              >
                {tasks.map((t) => (
                  <option key={t.path} value={t.path}>
                    {t.name} ({t.source})
                  </option>
                ))}
              </select>
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Provider preset</span>
              <select
                defaultValue=""
                onChange={(e) => applyPreset(e.target.value)}
                className="input-control w-full px-3 py-2"
              >
                <option value="">Choose preset (DeepSeek, Sarvam, OpenAI...)</option>
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.models[0]?.id ?? "custom"})
                  </option>
                ))}
              </select>
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Model ID</span>
              <input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="deepseek-flash or glm5.3-flash"
                className="input-control font-mono-num w-full px-3 py-2"
              />
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Base URL override (optional)</span>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.deepseek.com"
                className="input-control font-mono-num w-full px-3 py-2"
              />
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Judge model (optional)</span>
              <input
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
                placeholder="mock or gpt-4o-mini"
                className="input-control font-mono-num w-full px-3 py-2"
              />
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Fallback judge (optional)</span>
              <input
                value={fallbackJudgeModel}
                onChange={(e) => setFallbackJudgeModel(e.target.value)}
                placeholder="mock"
                className="input-control font-mono-num w-full px-3 py-2"
              />
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Hard USD cap (--max-cost-usd)</span>
              <input
                type="number"
                step="0.05"
                min="0"
                value={maxCostUsd}
                onChange={(e) => setMaxCostUsd(e.target.value)}
                className="input-control font-mono-num w-full px-3 py-2"
              />
            </label>

            <div className="grid grid-cols-3 gap-2.5">
              <label className="text-ui-sm block space-y-1.5">
                <span style={{ color: "var(--text-secondary)" }}>Limit</span>
                <input
                  type="number"
                  min="1"
                  value={limit}
                  onChange={(e) => setLimit(e.target.value)}
                  placeholder="All"
                  className="input-control font-mono-num w-full px-2.5 py-2"
                />
              </label>
              <label className="text-ui-sm block space-y-1.5">
                <span style={{ color: "var(--text-secondary)" }}>Repeats</span>
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={repeats}
                  onChange={(e) => setRepeats(Number(e.target.value) || 1)}
                  className="input-control font-mono-num w-full px-2.5 py-2"
                />
              </label>
              <label className="text-ui-sm block space-y-1.5">
                <span style={{ color: "var(--text-secondary)" }}>Bootstrap</span>
                <input
                  type="number"
                  min="0"
                  step="100"
                  value={bootstrapIters}
                  onChange={(e) => setBootstrapIters(Number(e.target.value) || 0)}
                  className="input-control font-mono-num w-full px-2.5 py-2"
                />
              </label>
            </div>
          </div>

          <div
            className="text-ui-sm mt-5 flex flex-wrap items-center gap-6 border-t pt-4"
            style={{ borderColor: "var(--border-subtle)" }}
          >
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={olmesEnabled}
                onChange={(e) => setOlmesEnabled(e.target.checked)}
                className="h-4 w-4 accent-[var(--accent-cyan)]"
              />
              <span>OLMES dual Cloze + MCF</span>
            </label>
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={enableThinking}
                onChange={(e) => setEnableThinking(e.target.checked)}
                className="h-4 w-4 accent-[var(--accent-cyan)]"
              />
              <span>Thinking trace extraction</span>
            </label>
            {enableThinking && (
              <input
                value={thinkEndToken}
                onChange={(e) => setThinkEndToken(e.target.value)}
                aria-label="Think end delimiter"
                className="input-control font-mono-num w-32 px-2.5 py-1"
              />
            )}
          </div>

          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => void handleEstimate()}
              className="telemetry-transition text-ui-base inline-flex cursor-pointer items-center gap-2 rounded-md border px-4 py-2 font-medium hover:bg-[var(--bg-elevated)]"
              style={{ borderColor: "var(--border-subtle)", color: "var(--text-primary)" }}
            >
              <Calculator className="h-4 w-4" /> Estimate cost
            </button>
            <button
              type="button"
              onClick={() => void handleLaunch()}
              disabled={runStatus === "running"}
              className="telemetry-transition text-ui-base inline-flex cursor-pointer items-center gap-2 rounded-md px-5 py-2 font-semibold hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
            >
              <Play className="h-4 w-4" /> Launch evaluation
            </button>
          </div>
        </Card>

        <Card
          title="Pre-flight cost ledger"
          subtitle="Target and judge token projection"
          padding="none"
          className="lg:col-span-4"
        >
          {estimate ? (
            <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="text-ui-sm flex justify-between px-5 py-3">
                <span style={{ color: "var(--text-secondary)" }}>Samples and repeats</span>
                <span className="font-mono-num font-semibold">
                  {estimate.n_samples} × {estimate.repeats}
                </span>
              </div>
              <div className="text-ui-sm flex justify-between px-5 py-3">
                <span style={{ color: "var(--text-secondary)" }}>Target tokens (in / out)</span>
                <span className="font-mono-num">
                  {formatTokens(estimate.target_prompt_tokens)} /{" "}
                  {formatTokens(estimate.target_completion_tokens)}
                </span>
              </div>
              <div className="text-ui-sm flex justify-between px-5 py-3">
                <span style={{ color: "var(--text-secondary)" }}>Target estimated USD</span>
                <span className="font-mono-num">{formatUsd(estimate.target_usd)}</span>
              </div>
              <div className="text-ui-sm flex justify-between px-5 py-3">
                <span style={{ color: "var(--text-secondary)" }}>Judge estimated USD</span>
                <span className="font-mono-num">{formatUsd(estimate.judge_usd)}</span>
              </div>
              <div className="text-ui-base flex items-center justify-between px-5 py-3.5 font-semibold">
                <span>Projected total</span>
                <span className="font-mono-num" style={{ color: "var(--accent-cyan)" }}>
                  {formatUsd(estimate.total_usd)}
                </span>
              </div>
              {estimate.exceeds_max_cost && (
                <div
                  className="text-ui-sm flex items-center gap-2.5 px-5 py-3"
                  style={{
                    backgroundColor: "var(--accent-rose-soft)",
                    color: "var(--accent-rose)",
                  }}
                >
                  <ShieldAlert className="h-4 w-4 shrink-0" />
                  <span>Exceeds --max-cost-usd cap ({formatUsd(Number(maxCostUsd))}).</span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-ui-sm p-5" style={{ color: "var(--text-secondary)" }}>
              Select Estimate cost to compute exact token and USD projections before dispatching inference requests.
            </div>
          )}
        </Card>
      </div>

      <Card
        title="SSE telemetry stream"
        subtitle="Real-time sample progress, reasoning snippets, and cumulative USD meter"
        padding="none"
        actions={
          <div className="flex items-center gap-4">
            <StatusBadge status={runStatus} />
            <span className="font-mono-num text-ui-sm">
              {completedSamples}/{totalSamples} ({pct}%) · {formatUsd(spentUsd)}
            </span>
          </div>
        }
      >
        <div
          className="h-[2px] w-full"
          style={{ backgroundColor: "var(--bg-elevated)" }}
        >
          <div
            className="h-full transition-[width] duration-200"
            style={{ width: `${pct}%`, backgroundColor: "var(--accent-cyan)" }}
          />
        </div>
        <div
          className="font-mono-num text-ui-sm max-h-72 overflow-y-auto divide-y"
          style={{
            backgroundColor: "var(--bg-code)",
            borderColor: "var(--border-subtle)",
          }}
        >
          {events.length === 0 ? (
            <div
              className="flex items-center gap-2 px-5 py-4"
              style={{ color: "var(--text-muted)" }}
            >
              <Terminal className="h-4 w-4" /> Waiting for SSE stream events...
            </div>
          ) : (
            events.map((ev, i) => (
              <div key={i} className="px-5 py-2">
                <span style={{ color: "var(--accent-cyan)" }}>[{ev.event}]</span>{" "}
                <span style={{ color: "var(--text-secondary)" }}>
                  {JSON.stringify(ev).slice(0, 220)}
                </span>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
