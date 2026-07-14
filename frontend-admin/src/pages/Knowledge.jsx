import { useCallback, useEffect, useState } from "react";
import { deleteKnowledge, listKnowledge, rebuildKnowledge, uploadKnowledge } from "../api";

export default function KnowledgePage() {
  const [files, setFiles] = useState([]);
  const [indexed, setIndexed] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await listKnowledge();
      setFiles(data.files || []);
      setIndexed(data.indexed_docs || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      await uploadKnowledge(file);
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function onRebuild() {
    if (!window.confirm("全量重建 Chroma + BM25 索引？")) return;
    setRebuilding(true);
    setError("");
    try {
      await rebuildKnowledge();
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setRebuilding(false);
    }
  }

  async function onDelete(filename) {
    if (!window.confirm(`删除 ${filename} 并重建索引？`)) return;
    setError("");
    try {
      await deleteKnowledge(filename);
      await load();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div>
      <header className="page-header">
        <h2>知识库管理</h2>
        <div className="header-actions">
          <button type="button" onClick={onRebuild} disabled={rebuilding}>
            {rebuilding ? "重建中…" : "重建索引"}
          </button>
          <label className="upload-btn">
            {uploading ? "上传中…" : "上传文档"}
            <input type="file" accept=".txt,.md" onChange={onUpload} hidden />
          </label>
        </div>
      </header>
      {error && <p className="error">{error}</p>}
      {loading ? (
        <p>加载中…</p>
      ) : (
        <>
          <section>
            <h3>磁盘文件 ({files.length})</h3>
            <table>
              <thead>
                <tr>
                  <th>文件名</th>
                  <th>大小</th>
                  <th>修改时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.filename}>
                    <td>{f.filename}</td>
                    <td>{(f.size_bytes / 1024).toFixed(1)} KB</td>
                    <td>{f.modified_at}</td>
                    <td>
                      <button type="button" className="danger" onClick={() => onDelete(f.filename)}>
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section>
            <h3>索引记录</h3>
            <ul className="indexed-list">
              {indexed.map((d) => (
                <li key={`${d.filename}-${d.indexed_at}`}>
                  {d.filename} — {d.chunk_count} chunks @ {d.indexed_at}
                </li>
              ))}
              {indexed.length === 0 && <li>暂无索引记录</li>}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
