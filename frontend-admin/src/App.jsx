import { NavLink, Route, Routes } from "react-router-dom";
import KnowledgePage from "./pages/Knowledge";
import ComplaintsPage from "./pages/Complaints";
import DashboardPage from "./pages/Dashboard";
import MetricsPage from "./pages/Metrics";
import InferencePage from "./pages/Inference";
import SettingsBar from "./components/SettingsBar";

export default function App() {
  return (
    <div className="admin-layout">
      <aside className="admin-sidebar">
        <h1>客服管理</h1>
        <nav>
          <NavLink to="/" end>仪表盘</NavLink>
          <NavLink to="/knowledge">知识库</NavLink>
          <NavLink to="/complaints">投诉工单</NavLink>
          <NavLink to="/metrics">指标</NavLink>
          <NavLink to="/inference">推理测试</NavLink>
          <a href="/" target="_blank" rel="noreferrer">聊天前台 ↗</a>
        </nav>
        <SettingsBar />
      </aside>
      <main className="admin-main">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/complaints" element={<ComplaintsPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="/inference" element={<InferencePage />} />
        </Routes>
      </main>
    </div>
  );
}
