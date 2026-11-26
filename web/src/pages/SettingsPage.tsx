import { KeyRound, PlugZap, Save, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card, ErrorBanner, StatusBadge } from "../components/common";
import type {
  ByokStatusResponse,
  ProviderPreset,
  SecretRole,
  TestConnectionResponse,
} from "../types/evalStudio";
import { formatBytes } from "../utils/format";

export function SettingsPage() {
  const [status, setStatus] = useState<ByokStatusResponse | null>(null);
  const [presets, setPresets] = useState<ProviderPreset[]>([]);
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({});
  const [storageMode, setStorageMode] = useState<"session" | "keyring">("session");
  const [pingResult, setPingResult] = useState<TestConnectionResponse | null>(null);
  const [pingModel, setPingModel] = useState("mock-deterministic");
  const [pingBackend, setPingBackend] = useState<"mock" | "http-byok">("mock");
  const [pingBaseUrl, setPingBaseUrl] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [s, p] = await Promise.all([api.getByokStatus(), api.getByokPresets()]);
      setStatus(s);
      setPresets(p.presets);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load BYOK settings.");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const handleSaveKey = async (role: SecretRole) => {
    const val = (keyInputs[role] ?? "").trim();
    if (!val) return;
    setError(null);
    try {
      await api.saveByokKey({ role, api_key: val, storage: storageMode });
      setKeyInputs((prev) => ({ ...prev, [role]: "" }));
      await refresh();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save key.");
    }
  };

  const handleDeleteKey = async (role: SecretRole) => {
    setError(null);
    try {
      await api.deleteByokKey(role, "all");
      await refresh();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to clear key.");
    }
  };

  const handlePing = async () => {
    setError(null);
    setPingResult(null);
    try {
      const res = await api.testByokConnection({
        backend: pingBackend,
        model_id: pingModel,
        base_url: pingBaseUrl.trim() || undefined,
      });
      setPingResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Connection test failed.");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1
          className="text-[26px] font-semibold tracking-[-0.03em]"
          style={{ color: "var(--text-primary)" }}
        >
          Provider credentials and local cache
        </h1>
        <p className="text-ui-base mt-1 max-w-[64ch]" style={{ color: "var(--text-secondary)" }}>
          Configure role-separated API credentials, verify gateway connectivity, and monitor the crash-safe SQLite and WAL cache.
        </p>
      </div>

      {error && <ErrorBanner error={error} />}

      <Card
        title="Role-separated BYOK credentials"
        subtitle="Credentials are stored in session memory or OS keyring; raw secrets are never returned over HTTP"
        padding="none"
        actions={
          <div className="flex items-center gap-3">
            <span className="text-ui-sm" style={{ color: "var(--text-secondary)" }}>
              Storage
            </span>
            <div
              role="radiogroup"
              aria-label="Key storage destination"
              className="inline-flex divide-x overflow-hidden rounded-md border"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              {(
                [
                  { id: "session", label: "Session memory" },
                  { id: "keyring", label: "OS keyring" },
                ] as const
              ).map((opt) => {
                const active = storageMode === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    onClick={() => setStorageMode(opt.id)}
                    className="telemetry-transition text-ui-sm cursor-pointer px-3 py-1 font-medium"
                    style={{
                      backgroundColor: active ? "var(--accent-cyan-soft)" : "var(--bg-elevated)",
                      color: active ? "var(--accent-cyan)" : "var(--text-secondary)",
                    }}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </div>
        }
      >
        <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
          {status?.roles.map((r) => (
            <div
              key={r.role}
              className="flex flex-col gap-3 px-5 py-4 md:flex-row md:items-center md:justify-between"
            >
              <div className="min-w-0 md:w-72">
                <div className="flex items-center gap-2">
                  <KeyRound className="h-4 w-4 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                  <span className="font-mono-num text-ui-sm font-semibold">{r.role}</span>
                </div>
                <div className="font-mono-num text-ui-sm mt-1">
                  <span
                    style={{
                      color: r.configured ? "var(--accent-emerald)" : "var(--text-muted)",
                    }}
                  >
                    {r.configured ? `${r.source} (${r.masked_hint ?? "***"})` : "unset"}
                  </span>
                </div>
              </div>

              <div className="flex flex-1 items-center gap-2.5 md:max-w-lg">
                <input
                  type="password"
                  value={keyInputs[r.role] ?? ""}
                  onChange={(e) =>
                    setKeyInputs((prev) => ({ ...prev, [r.role]: e.target.value }))
                  }
                  placeholder="Enter API key..."
                  className="input-control font-mono-num min-w-0 flex-1 px-3 py-1.5"
                />
                <button
                  type="button"
                  onClick={() => void handleSaveKey(r.role)}
                  className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md px-3.5 py-1.5 font-semibold hover:opacity-90"
                  style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
                >
                  <Save className="h-4 w-4" /> Save
                </button>
                {r.configured && (
                  <button
                    type="button"
                    onClick={() => void handleDeleteKey(r.role)}
                    className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1 rounded-md border px-2.5 py-1.5 hover:bg-[var(--bg-elevated)]"
                    style={{ borderColor: "var(--border-subtle)", color: "var(--accent-rose)" }}
                    title="Clear key"
                    aria-label={`Clear ${r.role}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Connection health check" subtitle="Ping MockGateway or live HttpBYOKGateway">
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div
                role="radiogroup"
                aria-label="Ping backend"
                className="inline-flex divide-x overflow-hidden rounded-md border"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                {(["mock", "http-byok"] as const).map((opt) => {
                  const active = pingBackend === opt;
                  return (
                    <button
                      key={opt}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      onClick={() => setPingBackend(opt)}
                      className="telemetry-transition text-ui-sm flex-1 cursor-pointer px-3 py-2 font-medium"
                      style={{
                        backgroundColor: active ? "var(--accent-cyan-soft)" : "var(--bg-elevated)",
                        color: active ? "var(--accent-cyan)" : "var(--text-secondary)",
                      }}
                    >
                      {opt === "mock" ? "Mock offline" : "Live HTTP"}
                    </button>
                  );
                })}
              </div>
              <input
                value={pingModel}
                onChange={(e) => setPingModel(e.target.value)}
                placeholder="deepseek-flash"
                className="input-control font-mono-num px-3 py-2"
              />
            </div>

            <input
              value={pingBaseUrl}
              onChange={(e) => setPingBaseUrl(e.target.value)}
              placeholder="Base URL (optional, e.g. https://api.deepseek.com)"
              className="input-control font-mono-num w-full px-3 py-2"
            />

            <button
              type="button"
              onClick={() => void handlePing()}
              className="telemetry-transition text-ui-base inline-flex cursor-pointer items-center gap-2 rounded-md px-4 py-2 font-semibold hover:opacity-90"
              style={{ backgroundColor: "var(--accent-emerald)", color: "var(--text-inverse)" }}
            >
              <PlugZap className="h-4 w-4" /> Test connection
            </button>

            {pingResult && (
              <div
                className="border-t pt-3"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <div className="flex items-center justify-between">
                  <StatusBadge status={pingResult.ok ? "VERIFIED" : "FAILED"} />
                  <span className="font-mono-num text-ui-sm">
                    {pingResult.latency_ms.toFixed(1)} ms
                  </span>
                </div>
                <div
                  className="font-mono-num text-ui-sm mt-2"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {pingResult.ok ? pingResult.reply_preview : pingResult.error}
                </div>
              </div>
            )}
          </div>
        </Card>

        <Card
          title="SQLite and WAL response cache"
          subtitle="User-owned crash-safe fingerprint store"
          padding="none"
        >
          {status ? (
            <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="text-ui-sm flex items-center justify-between gap-4 px-5 py-3.5">
                <span style={{ color: "var(--text-secondary)" }}>Cache directory</span>
                <span className="font-mono-num max-w-xs truncate">{status.cache_dir}</span>
              </div>
              <div className="text-ui-sm flex items-center justify-between px-5 py-3.5">
                <span style={{ color: "var(--text-secondary)" }}>SQLite database size</span>
                <span className="font-mono-num">{formatBytes(status.cache_stats.db_bytes)}</span>
              </div>
              <div className="text-ui-sm flex items-center justify-between px-5 py-3.5">
                <span style={{ color: "var(--text-secondary)" }}>WAL entries</span>
                <span className="font-mono-num">{status.cache_stats.wal_entries}</span>
              </div>
            </div>
          ) : null}
        </Card>
      </div>

      <Card
        title="Built-in provider presets"
        subtitle="Supported endpoints for DeepSeek, Sarvam V2, OpenAI, Anthropic, Gemini, OpenRouter, and Ollama"
        padding="none"
      >
        <div className="overflow-x-auto">
          <table className="text-ui-sm w-full text-left">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                  Provider
                </th>
                <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                  Default base URL
                </th>
                <th className="px-4 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                  Auth header
                </th>
                <th className="px-5 py-3 font-medium" style={{ color: "var(--text-secondary)" }}>
                  Models
                </th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
              {presets.map((p) => (
                <tr key={p.id}>
                  <td className="px-5 py-3 font-semibold">{p.name}</td>
                  <td
                    className="font-mono-num px-4 py-3"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {p.default_base_url || "offline"}
                  </td>
                  <td className="font-mono-num px-4 py-3" style={{ color: "var(--text-muted)" }}>
                    {p.auth_header}
                  </td>
                  <td className="font-mono-num px-5 py-3" style={{ color: "var(--accent-cyan)" }}>
                    {p.models.map((m) => m.id).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
