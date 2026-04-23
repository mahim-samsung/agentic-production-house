"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Download, FileJson, Loader2 } from "lucide-react";
import { PLATFORMS } from "@/lib/constants";
import { loadAdminSettings } from "@/lib/settings";
import styles from "./production.module.css";

type JobStatus = "idle" | "running" | "complete" | "failed";

type PollResult = {
  status: JobStatus;
  result?: {
    ok?: boolean;
    error?: string;
    traceback?: string;
    output_path?: string;
    duration?: number;
    resolution?: string;
    processing_time_seconds?: number;
    title?: string;
  };
};

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [platform, setPlatform] = useState("youtube_long");
  const [outputFilename, setOutputFilename] = useState("");
  const [skipAudio, setSkipAudio] = useState(false);
  const [generateMusic, setGenerateMusic] = useState(false);
  const [keepSourceAudio, setKeepSourceAudio] = useState(false);

  const [mediaFiles, setMediaFiles] = useState<File[]>([]);
  const [musicFiles, setMusicFiles] = useState<File[]>([]);

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus>("idle");
  const [pollResult, setPollResult] = useState<PollResult["result"] | undefined>();
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const adminRef = useRef(loadAdminSettings());

  useEffect(() => {
    const sync = () => {
      adminRef.current = loadAdminSettings();
    };
    window.addEventListener("focus", sync);
    return () => window.removeEventListener("focus", sync);
  }, []);

  const videoUrl = useMemo(
    () => (jobId && jobStatus === "complete" ? `/api/production/${jobId}/video` : null),
    [jobId, jobStatus]
  );

  const poll = useCallback(async (id: string) => {
    const r = await fetch(`/api/production/${id}`, { cache: "no-store" });
    const raw = await r.text();
    let data: PollResult;
    try {
      data = JSON.parse(raw) as PollResult;
    } catch {
      throw new Error("Bad response from status API");
    }
    if (!r.ok) {
      const errBody = data as unknown as { error?: string };
      throw new Error(typeof errBody.error === "string" ? errBody.error : r.statusText);
    }
    if (data.status === "complete") {
      setJobStatus("complete");
      setPollResult(data.result);
      return true;
    }
    if (data.status === "failed") {
      setJobStatus("failed");
      setPollResult(data.result);
      return true;
    }
    return false;
  }, []);

  const [workElapsedSec, setWorkElapsedSec] = useState(0);

  useEffect(() => {
    if (jobStatus !== "running" || !jobId) {
      setWorkElapsedSec(0);
      return;
    }
    const t0 = Date.now();
    const id = window.setInterval(() => setWorkElapsedSec(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [jobId, jobStatus]);

  useEffect(() => {
    if (!jobId || jobStatus !== "running") return;
    let cancelled = false;
    let delayMs = 2500;
    const maxDelayMs = 12000;
    let timeoutId: ReturnType<typeof setTimeout>;
    let pollFailures = 0;

    const schedule = (ms: number) => {
      timeoutId = setTimeout(tick, ms);
    };

    const tick = async () => {
      if (cancelled) return;
      if (typeof document !== "undefined" && document.hidden) {
        schedule(10000);
        return;
      }
      let done = false;
      try {
        done = await poll(jobId);
        pollFailures = 0;
      } catch (e) {
        pollFailures += 1;
        if (pollFailures >= 8) {
          setJobStatus("failed");
          setPollResult({
            ok: false,
            error: e instanceof Error ? e.message : "Status check failed repeatedly; the job may still be running on the server.",
          });
          return;
        }
      }
      if (cancelled || done) return;
      delayMs = Math.min(maxDelayMs, Math.round(delayMs * 1.35));
      schedule(delayMs);
    };

    schedule(delayMs);
    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [jobId, jobStatus, poll]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    adminRef.current = loadAdminSettings();
    const a = adminRef.current;

    setSubmitError(null);
    setBusy(true);
    setPollResult(undefined);
    setJobId(null);
    setJobStatus("running");

    const fd = new FormData();
    fd.set("prompt", prompt);
    fd.set("platform", platform);
    fd.set("outputFilename", outputFilename.trim());
    fd.set("skipAudio", skipAudio ? "true" : "false");
    fd.set("generateMusic", generateMusic ? "true" : "false");
    fd.set("keepSourceAudio", keepSourceAudio ? "true" : "false");
    fd.set("writerConstrained", a.writerConstrained ? "true" : "false");
    fd.set("visionBackend", a.visionBackend);
    fd.set("vlmSemanticsBackend", a.vlmSemanticsBackend);
    if (a.vlmSemanticsBackend === "qwen2_5_vl" && a.qwen25vlModelId.trim()) {
      fd.set("qwen25vlModelId", a.qwen25vlModelId.trim());
    }
    if (a.ollamaModel.trim()) fd.set("ollamaModel", a.ollamaModel.trim());
    if (a.ollamaBaseUrl.trim()) fd.set("ollamaBaseUrl", a.ollamaBaseUrl.trim());
    if (a.configServerPath.trim()) fd.set("configServerPath", a.configServerPath.trim());
    fd.set("videoMomentBackend", a.videoMomentBackend);
    if (a.videoMomentBackend === "internvideo2") {
      fd.set("internvideo2Enabled", a.internvideo2Enabled ? "true" : "false");
      if (a.internvideo2ModelId.trim()) fd.set("internvideo2ModelId", a.internvideo2ModelId.trim());
    }
    for (const f of mediaFiles) fd.append("media", f);
    for (const f of musicFiles) fd.append("music", f);

    try {
      const res = await fetch("/api/production", { method: "POST", body: fd });
      const json = await res.json();
      if (!res.ok) {
        setJobStatus("failed");
        setSubmitError(json.error || res.statusText);
        return;
      }
      setJobId(json.jobId);
      const pr = await fetch(`/api/production/${json.jobId}`, { cache: "no-store" });
      const data = (await pr.json()) as PollResult;
      if (data.status === "complete") {
        setJobStatus("complete");
        setPollResult(data.result);
      } else if (data.status === "failed") {
        setJobStatus("failed");
        setPollResult(data.result);
      } else {
        setJobStatus("running");
      }
    } catch (err) {
      setJobStatus("failed");
      setSubmitError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.shell}>
      <form onSubmit={onSubmit} className={styles.grid}>
        <section className={styles.card}>
          <label className={styles.drop}>
            <input
              type="file"
              multiple
              accept="video/*,image/*"
              style={{ display: "none" }}
              onChange={(e) => setMediaFiles(Array.from(e.target.files || []))}
            />
            {mediaFiles.length ? (
              <span className={styles.mono}>{mediaFiles.map((f) => f.name).join(" · ")}</span>
            ) : (
              <>Video / image</>
            )}
          </label>

          <label className={styles.drop} style={{ marginTop: "0.65rem" }}>
            <input
              type="file"
              multiple
              accept="audio/*,.mp3,.wav,.aac,.flac,.ogg,.m4a"
              style={{ display: "none" }}
              onChange={(e) => setMusicFiles(Array.from(e.target.files || []))}
            />
            {musicFiles.length ? (
              <span className={styles.mono}>{musicFiles.map((f) => f.name).join(" · ")}</span>
            ) : (
              <>Music · optional</>
            )}
          </label>
        </section>

        <section className={styles.card}>
          <textarea
            className={styles.textarea}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Prompt"
            required
          />

          <div className={`${styles.grid} ${styles.grid2}`} style={{ marginTop: "0.85rem" }}>
            <select className={styles.select} value={platform} onChange={(e) => setPlatform(e.target.value)}>
              {PLATFORMS.map((p) => (
                <option key={p.id} value={p.id} title={p.hint}>
                  {p.label}
                </option>
              ))}
            </select>
            <input
              className={styles.input}
              value={outputFilename}
              onChange={(e) => setOutputFilename(e.target.value)}
              placeholder="output.mp4"
            />
          </div>
        </section>

        <section className={styles.card}>
          <div className={styles.row}>
            <label className={styles.check}>
              <input
                type="checkbox"
                checked={skipAudio}
                onChange={(e) => {
                  const v = e.target.checked;
                  setSkipAudio(v);
                  if (v) setGenerateMusic(false);
                }}
              />
              No audio pass
            </label>
            <label
              className={styles.check}
              style={{ opacity: skipAudio ? 0.55 : 1 }}
              title={
                skipAudio
                  ? "No audio pass skips the whole audio step, so MusicGen cannot run. Turn it off to use MusicGen."
                  : "When the audio pass runs, try AI background music from the creative brief and your prompt."
              }
            >
              <input
                type="checkbox"
                checked={generateMusic}
                disabled={skipAudio}
                onChange={(e) => setGenerateMusic(e.target.checked)}
              />
              MusicGen
            </label>
          </div>
          <label className={styles.check} style={{ marginTop: "0.65rem", display: "flex" }}>
            <input
              type="checkbox"
              checked={keepSourceAudio}
              onChange={(e) => setKeepSourceAudio(e.target.checked)}
            />
            <span title="When off, clip microphone/camera audio is not used in the edit; BGM can still be added if the audio pass runs.">
              Keep original clip audio
            </span>
          </label>
          {skipAudio && (
            <p style={{ margin: "0.55rem 0 0", fontSize: "0.82rem", color: "var(--muted)" }}>
              No audio pass skips normalize, fades, and all BGM (including MusicGen).
            </p>
          )}
        </section>

        <div className={styles.actions}>
          <button type="submit" className={`${styles.btn} ${styles.btnPrimary}`} disabled={busy || !mediaFiles.length}>
            {busy ? (
              <>
                <Loader2
                  size={16}
                  style={{ display: "inline", marginRight: 8, animation: "spin 0.9s linear infinite" }}
                />
                …
              </>
            ) : (
              "Run"
            )}
          </button>
          <button
            type="button"
            className={`${styles.btn} ${styles.btnGhost}`}
            onClick={() => {
              setJobId(null);
              setJobStatus("idle");
              setPollResult(undefined);
              setSubmitError(null);
            }}
          >
            Clear
          </button>
        </div>
      </form>

      {(submitError || jobStatus !== "idle") && (
        <div
          className={`${styles.status} ${
            jobStatus === "complete" ? styles.statusOk : jobStatus === "failed" ? styles.statusErr : ""
          }`}
        >
          {submitError && <p style={{ margin: "0 0 0.5rem", color: "var(--danger)" }}>{submitError}</p>}
          {jobId && <p className={styles.mono} style={{ margin: "0 0 0.35rem" }}>{jobId}</p>}
          {jobStatus === "running" && (
            <p style={{ margin: 0, color: "var(--muted)" }}>
              <Loader2 size={16} style={{ display: "inline", marginRight: 8, animation: "spin 0.9s linear infinite" }} />
              Working
              {workElapsedSec > 0 ? ` · ${workElapsedSec}s` : ""}
              <span style={{ display: "block", marginTop: "0.35rem", fontSize: "0.85rem" }}>
                First model / vision steps can take several minutes on CPU. If this never finishes, open the job folder under{" "}
                <span className={styles.mono}>.tmp/web-jobs/</span> and check <span className={styles.mono}>worker.log</span> and{" "}
                <span className={styles.mono}>result.json</span>.
              </span>
            </p>
          )}
          {jobStatus === "complete" && pollResult?.ok !== false && (
            <>
              {pollResult?.title && <p style={{ margin: "0 0 0.35rem" }}>{pollResult.title}</p>}
              <p className={styles.mono} style={{ margin: 0 }}>
                {pollResult?.duration != null && `${pollResult.duration.toFixed(1)}s`}
                {pollResult?.resolution && ` · ${pollResult.resolution}`}
                {pollResult?.processing_time_seconds != null && ` · ${pollResult.processing_time_seconds.toFixed(0)}s`}
              </p>
              <div className={styles.links}>
                {videoUrl && (
                  <a className={styles.linkBtn} href={videoUrl} download>
                    <Download size={14} /> MP4
                  </a>
                )}
                {jobId && (
                  <a className={styles.linkBtn} href={`/api/production/${jobId}/report`} target="_blank" rel="noreferrer">
                    <FileJson size={14} /> JSON
                  </a>
                )}
              </div>
              {videoUrl && <video className={styles.video} src={videoUrl} controls playsInline />}
            </>
          )}
          {(jobStatus === "failed" || pollResult?.ok === false) && pollResult && (
            <>
              <p style={{ margin: "0 0 0.5rem", color: "var(--danger)" }}>{pollResult.error || "Failed"}</p>
              {pollResult.traceback && (
                <pre
                  style={{
                    margin: 0,
                    maxHeight: 220,
                    overflow: "auto",
                    fontSize: "0.72rem",
                    color: "var(--muted-2)",
                  }}
                >
                  {pollResult.traceback}
                </pre>
              )}
            </>
          )}
        </div>
      )}

      <footer className={styles.footer}>Developed by Mahbub Islam Mahim - Copyright ST-Web Team 2026
      </footer>

      <style jsx global>{`
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
