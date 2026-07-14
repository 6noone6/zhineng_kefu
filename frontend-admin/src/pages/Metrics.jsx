import { useCallback, useEffect, useState } from "react";
import { fetchMetricsText, parsePrometheusMetrics } from "../api";

const METRIC_LABELS = {
  kefu_chat_requests_total: "聊天请求总数",
  kefu_sessions_created_total: "新建会话",
  kefu_tool_calls_total: "工具调用",
  kefu_rag_queries_total: "RAG 检索",
  kefu_complaints_recorded_total: "投诉记录",
  kefu_knowledge_uploads_total: "知识库上传",
  kefu_kimi_tokens_total: "Kimi Token",
};

export default function MetricsPage() {
  const [metrics, setMetrics] = useState({});
  const [raw, setRaw] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const text = await fetchMetricsText();
      setRaw(text);
      setMetrics(parsePrometheusMetrics(text));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const summary = Object.entries(METRIC_LABELS)
    .map(([key, label]) => ({ key, label, value: metrics[key] }))
    .filter((m) => m.value !== undefined);

  return (
    <div>
      <header className="page-header">
        <h2>Prometheus 指标</h2>
        <button type="button" onClick={load}>
          刷新
        </button>
      </header>
      {error && <p className="error">{error}</p>}
      {loading ? (
        <p>加载中…</p>
      ) : (
        <>
          <section className="card-grid">
            {summary.map((m) => (
              <div key={m.key} className="card">
                <h3>{m.label}</h3>
                <p className="metric-value">{m.value?.toLocaleString() ?? 0}</p>
                <p className="muted">{m.key}</p>
              </div>
            ))}
            {summary.length === 0 && <p>暂无指标数据（服务尚无请求）</p>}
          </section>
          <section>
            <h3>原始指标</h3>
            <pre className="metrics-raw">{raw.slice(0, 4000)}{raw.length > 4000 ? "\n…" : ""}</pre>
          </section>
        </>
      )}
    </div>
  );
}
