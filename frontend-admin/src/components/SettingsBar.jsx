import { useState } from "react";
import { getApiKey, setApiKey } from "../api";

export default function SettingsBar() {
  const [key, setKey] = useState(getApiKey());
  const [saved, setSaved] = useState(false);

  function save() {
    setApiKey(key.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="settings-bar">
      <label>
        API Key
        <input
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder="X-API-Key"
        />
      </label>
      <button type="button" onClick={save}>
        保存
      </button>
      {saved && <span className="hint">已保存</span>}
    </div>
  );
}
