import { KeyRound, PlugZap, Save, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Badge, Card, ErrorBanner } from "../components/common";
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
      {error && <ErrorBanner error={error} />}

      {/* Role-Separated API Keys */}
      <Card
        title="Role-Separated BYOK Credentials"
        subtitle="Keys are stored in session memory or OS keyring — raw secrets are never echoed over HTTP"
        actions={
          <div className="flex items-center gap-2 text-xs">
            <span style={{ color: "var(--text-secondary)" }}>Storage:</span>
            <select
              value={storageMode}
              onChange={(e) => setStorageMode(e.target.value as "session" | "keyring")}
              className="rounded-lg border px-2.5 py-1 text-xs"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
            >
              <option value="session">Session Memory</option>
              <option value="keyring">OS Keyring</option>
            </select>
          </div>
        }
      >
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {status?.roles.map((r) => (
            <div
              key={r.role}
              className="space-y-3 rounded-xl border p-4"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <KeyRound className="h-4 w-4" style={{ color: "var(--accent-cyan)" }} />
                  <span className="font-mono text-xs font-semibold">{r.role}</span>
                </div>
                <Badge variant={r.configured ? "emerald" : "neutral"}>
                  {r.configured ? `${r.source} · ${r.masked_hint ?? "***"}` : "unset"}
                </Badge>
              </div>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={keyInputs[r.role] ?? ""}
                  onChange={(e) =>
                    setKeyInputs((prev) => ({ ...prev, [r.role]: e.target.value }))
                  }
                  placeholder="Enter API key..."
                  className="min-w-0 flex-1 rounded-lg border px-3 py-1.5 font-mono text-xs"
                  style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-canvas)" }}
                />
                <button
                  type="button"
                  onClick={() => void handleSaveKey(r.role)}
                  className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-semibold"
                  style={{ backgroundColor: "var(--accent-cyan)", color: "#090D16" }}
                >
                  <Save className="h-3.5 w-3.5" /> Save
                </button>
                {r.configured && (
                  <button
                    type="button"
                    onClick={() => void handleDeleteKey(r.role)}
                    className="inline-flex items-center rounded-lg border px-2.5 py-1.5 text-xs"
                    style={{ borderColor: "var(--border-subtle)", color: "var(--accent-rose)" }}
                    title="Clear key"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Connection Tester & Cache Telemetry */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Test Provider Connection" subtitle="Ping MockGateway or live HttpBYOKGateway">
          <div className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <select
                value={pingBackend}
                onChange={(e) => setPingBackend(e.target.value as "mock" | "http-byok")}
                className="rounded-lg border px-3 py-2"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              >
                <option value="mock">mock (offline)</option>
                <option value="http-byok">http-byok</option>
              </select>
              <input
                value={pingModel}
                onChange={(e) => setPingModel(e.target.value)}
                placeholder="deepseek-flash"
                className="rounded-lg border px-3 py-2 font-mono"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              />
            </div>
            <input
              value={pingBaseUrl}
              onChange={(e) => setPingBaseUrl(e.target.value)}
              placeholder="Base URL (optional, e.g. https://api.deepseek.com)"
              className="w-full rounded-lg border px-3 py-2 font-mono"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
            />
            <button
              type="button"
              onClick={() => void handlePing()}
              className="inline-flex items-center gap-1.5 rounded-lg px-4 py-2 font-semibold"
              style={{ backgroundColor: "var(--accent-emerald)", color: "#090D16" }}
            >
              <PlugZap className="h-3.5 w-3.5" /> Test Connection
            </button>
            {pingResult && (
              <div
                className="rounded-xl border p-3"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-canvas)" }}
              >
                <div className="flex items-center justify-between">
                  <Badge variant={pingResult.ok ? "emerald" : "rose"}>
                    {pingResult.ok ? "CONNECTED" : "FAILED"}
                  </Badge>
                  <span className="font-mono">{pingResult.latency_ms.toFixed(1)} ms</span>
                </div>
                <div className="mt-2 font-mono text-[11px]" style={{ color: "var(--text-secondary)" }}>
                  {pingResult.ok ? pingResult.reply_preview : pingResult.error}
                </div>
              </div>
            )}
          </div>
        </Card>

        <Card title="Local SQLite + WAL Cache" subtitle="User-owned crash-safe fingerprint cache" elevated>
          {status ? (
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>Cache Directory</span>
                <span className="max-w-xs truncate font-mono">{status.cache_dir}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>SQLite Size</span>
                <span className="font-mono">{formatBytes(status.cache_stats.db_bytes)}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-secondary)" }}>WAL Entries</span>
                <span className="font-mono">{status.cache_stats.wal_entries}</span>
              </div>
            </div>
          ) : null}
        </Card>
      </div>

      {/* Supported Provider Presets */}
      <Card title="Supported Provider Presets" subtitle="Built-in dialects for DeepSeek, Sarvam V2, OpenAI, Anthropic, Gemini, OpenRouter & Ollama">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {presets.map((p) => (
            <div
              key={p.id}
              className="rounded-xl border p-3 text-xs"
              style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
            >
              <div className="font-semibold">{p.name}</div>
              <div className="mt-0.5 truncate font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>
                {p.default_base_url || "offline"}
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {p.models.slice(0, 2).map((m) => (
                  <Badge key={m.id} variant="cyan">
                    {m.id}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
