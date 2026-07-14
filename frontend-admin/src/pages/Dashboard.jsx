import { useCallback, useEffect, useState } from "react";
import { fetchHealth, fetchHealthDeep } from "../api";

function StatusBadge({ status }) {
  const cls =
    status === "ok" || status === "configured"
      ? "badge-ok"
      : status === "degraded"
        ? "badge-warn"
        : "badge-down";
  return <span className={`status-badge ${cls}`}>{status}</span>;
}

export default function DashboardPage() {
  const [health, setHealth] = useState(null);
  const [deep, setDeep] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [h, d] = await Promise.all([fetchHealth(), fetchHealthDeep()]);
      setHealth(h);
      setDeep(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <header className="page-header">
        <h2>系统仪表盘</h2>
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
            <div className="card">
              <h3>服务状态</h3>
              <p>
                <StatusBadge status={health?.status} /> {health?.service}
              </p>
              <p className="muted">环境: {deep?.env}</p>
            </div>
            <div className="card">
              <h3>PostgreSQL</h3>
              <StatusBadge status={deep?.components?.postgres?.status} />
            </div>
            <div className="card">
              <h3>Redis</h3>
              <StatusBadge status={deep?.components?.redis?.status} />
            </div>
            <div className="card">
              <h3>Kimi API</h3>
              <StatusBadge status={deep?.components?.kimi?.status} />
            </div>
            <div className="card">
              <h3>RAG 后端</h3>
              <p>{deep?.components?.rag?.backend}</p>
              {deep?.components?.rag?.qwen_inference_url && (
                <p className="muted">Qwen: {deep.components.rag.qwen_inference_url}</p>
              )}
            </div>
            {deep?.components?.qwen_remote && (
              <div className="card">
                <h3>远程 Qwen</h3>
                <StatusBadge status={deep.components.qwen_remote.status} />
              </div>
            )}
          </section>
          <section>
            <h3>已实现能力</h3>
            <ul className="feature-list">
              <li>WebSocket / REST 流式对话 + ReAct 多步工具编排</li>
              <li>6 个 Agent 工具：知识库、订单、物流、退换货、投诉、我的订单</li>
              <li>Hybrid RAG（BM25 + Chroma + RRF）+ 多语言知识库</li>
              <li>JWT 登录 + 订单绑定 + 会话持久化</li>
              <li>知识库上传/重建 + 投诉工单管理</li>
              <li>Prometheus 指标（延迟、工具、Token 用量）</li>
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
