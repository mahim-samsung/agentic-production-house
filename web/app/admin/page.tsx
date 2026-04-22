"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  defaultAdminSettings,
  loadAdminSettings,
  saveAdminSettings,
  type AdminSettings,
} from "@/lib/settings";
import styles from "../production.module.css";

export default function AdminPage() {
  const [s, setS] = useState<AdminSettings>({ ...defaultAdminSettings });
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setS(loadAdminSettings());
  }, []);

  function update<K extends keyof AdminSettings>(key: K, value: AdminSettings[K]) {
    setS((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  }

  function onSave(e: React.FormEvent) {
    e.preventDefault();
    saveAdminSettings(s);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function onReset() {
    setS({ ...defaultAdminSettings });
    saveAdminSettings({ ...defaultAdminSettings });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className={styles.shell}>
      <h1 className={styles.title} style={{ marginBottom: "1.25rem" }}>
        Admin
      </h1>

      <form onSubmit={onSave} className={styles.grid}>
        <section className={styles.card}>
          <select
            className={styles.select}
            value={s.visionBackend}
            onChange={(e) => update("visionBackend", e.target.value as AdminSettings["visionBackend"])}
          >
            <option value="inherit">Vision · default</option>
            <option value="siglip2">siglip2</option>
            <option value="open_clip">open_clip</option>
          </select>
        </section>

        <section className={styles.card}>
          <input
            className={styles.input}
            style={{ marginBottom: "0.65rem" }}
            value={s.ollamaModel}
            onChange={(e) => update("ollamaModel", e.target.value)}
            placeholder="Ollama model override"
          />
          <input
            className={styles.input}
            value={s.ollamaBaseUrl}
            onChange={(e) => update("ollamaBaseUrl", e.target.value)}
            placeholder="Ollama URL · e.g. http://127.0.0.1:11434"
          />
        </section>

        <section className={styles.card}>
          <label className={styles.check} style={{ margin: 0 }}>
            <input
              type="checkbox"
              checked={s.writerConstrained}
              onChange={(e) => update("writerConstrained", e.target.checked)}
            />
            Constrained writer
          </label>
        </section>

        <section className={styles.card}>
          <select
            className={styles.select}
            value={s.videoMomentBackend}
            onChange={(e) =>
              update("videoMomentBackend", e.target.value as AdminSettings["videoMomentBackend"])
            }
          >
            <option value="inherit">Moments · yaml</option>
            <option value="heuristic">heuristic</option>
            <option value="internvideo2">internvideo2</option>
          </select>
          {s.videoMomentBackend === "internvideo2" && (
            <>
              <label className={styles.check} style={{ margin: "0.75rem 0 0" }}>
                <input
                  type="checkbox"
                  checked={s.internvideo2Enabled}
                  onChange={(e) => update("internvideo2Enabled", e.target.checked)}
                />
                InternVideo2 on
              </label>
              <input
                className={styles.input}
                style={{ marginTop: "0.65rem" }}
                value={s.internvideo2ModelId}
                onChange={(e) => update("internvideo2ModelId", e.target.value)}
                placeholder="InternVideo2 model id"
              />
            </>
          )}
        </section>

        <section className={styles.card}>
          <input
            className={styles.input}
            value={s.configServerPath}
            onChange={(e) => update("configServerPath", e.target.value)}
            placeholder="config.yaml path on this machine (optional)"
          />
        </section>

        <div className={styles.actions}>
          <button type="submit" className={`${styles.btn} ${styles.btnPrimary}`}>
            Save
          </button>
          <button type="button" className={`${styles.btn} ${styles.btnGhost}`} onClick={onReset}>
            Defaults
          </button>
        </div>
      </form>

      {saved && (
        <p style={{ marginTop: "0.85rem", fontSize: "0.82rem", color: "var(--success)" }}>Saved</p>
      )}

      <footer className={styles.footer}>
       Developed by Mahbub Islam Mahim - Copyright ST-Web Team 2026
      </footer>
    </div>
  );
}
