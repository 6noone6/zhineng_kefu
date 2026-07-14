import { useCallback, useEffect, useState } from "react";
import { listComplaints, updateComplaintStatus } from "../api";

const STATUSES = ["Received", "InReview", "Resolved"];

export default function ComplaintsPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await listComplaints();
      setItems(data.items || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function onStatusChange(id, status) {
    setError("");
    try {
      await updateComplaintStatus(id, status);
      await load();
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div>
      <header className="page-header">
        <h2>投诉工单看板</h2>
        <button type="button" onClick={load}>
          刷新
        </button>
      </header>
      {error && <p className="error">{error}</p>}
      {loading ? (
        <p>加载中…</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>工单号</th>
              <th>状态</th>
              <th>详情</th>
              <th>会话</th>
              <th>时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {items.map((c) => (
              <tr key={c.id}>
                <td>{c.ticket_id}</td>
                <td>
                  <span className={`badge badge-${c.status}`}>{c.status}</span>
                </td>
                <td className="details-cell">{c.details}</td>
                <td>{c.session_id?.slice(0, 8) || "—"}</td>
                <td>{c.created_at}</td>
                <td>
                  <select
                    value={c.status}
                    onChange={(e) => onStatusChange(c.id, e.target.value)}
                  >
                    {STATUSES.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr>
                <td colSpan={6}>暂无投诉工单</td>
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}
