"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { SearchResult, SearchResponse, StatusResponse, UploadResponse } from "@/lib/types";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8080";
const FRAME = 224;

export default function Page() {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("person in a red jacket");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const addLog = useCallback((l: string) => setLog((p) => [...p.slice(-40), l]), []);

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    if (f) setVideoSrc(URL.createObjectURL(f));
  };

  const poll = useCallback(
    (id: string) => {
      const tick = async () => {
        try {
          const r = await fetch(`${GATEWAY}/status/${id}`);
          const st: StatusResponse = await r.json();
          setStatus(st);
          if (!st.done) setTimeout(tick, 700);
          else addLog(`✓ processed ${st.frames} frames — ready to search`);
        } catch {
          /* gateway not up */
        }
      };
      tick();
    },
    [addLog]
  );

  const ingest = async () => {
    setBusy(true);
    setResults([]);
    setActiveIdx(null);
    try {
      const form = new FormData();
      if (file) form.append("video", file);
      const r = await fetch(`${GATEWAY}/upload`, { method: "POST", body: form });
      const data: UploadResponse = await r.json();
      setVideoId(data.video_id);
      addLog(`↑ uploaded → ${data.video_id} (${data.status})`);
      poll(data.video_id);
    } catch (err) {
      addLog(`⚠ upload failed — is the Go gateway running? (${String(err)})`);
    } finally {
      setBusy(false);
    }
  };

  const search = async () => {
    if (!videoId) {
      addLog("⚠ ingest a video first");
      return;
    }
    setBusy(true);
    try {
      const r = await fetch(`${GATEWAY}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId, query, top_k: 5 }),
      });
      const data: SearchResponse = await r.json();
      setResults(data.results || []);
      addLog(`🔍 "${query}" → ${data.results?.length ?? 0} hits`);
    } catch (err) {
      addLog(`⚠ search failed (${String(err)})`);
    } finally {
      setBusy(false);
    }
  };

  const jumpTo = (idx: number) => {
    const res = results[idx];
    setActiveIdx(idx);
    const v = videoRef.current;
    if (v && isFinite(res.timestamp_ms)) v.currentTime = res.timestamp_ms / 1000;
    drawBox(res);
  };

  const drawBox = useCallback((res: SearchResult | null) => {
    const c = canvasRef.current;
    const v = videoRef.current;
    if (!c || !v) return;
    c.width = v.clientWidth;
    c.height = v.clientHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!res?.box) return;
    const [x, y, w, h] = res.box;
    const sx = c.width / FRAME;
    const sy = c.height / FRAME;
    ctx.strokeStyle = "#35d07f";
    ctx.lineWidth = 2;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    ctx.fillStyle = "rgba(53,208,127,0.9)";
    ctx.font = "12px system-ui";
    ctx.fillText(res.labels[0] || "match", x * sx, Math.max(12, y * sy - 4));
  }, []);

  useEffect(() => {
    if (activeIdx != null) drawBox(results[activeIdx]);
  }, [activeIdx, results, drawBox]);

  const durationMs = Math.max(12000, ...results.map((r) => r.timestamp_ms + 1000), 1);

  return (
    <main style={{ maxWidth: 980, margin: "0 auto", padding: 24 }}>
      <h1 style={{ marginBottom: 2 }}>vidgrep</h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        Search <em>inside</em> video with natural language · Go ingestion · Rust decode · Python CLIP · TS UI
      </p>

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", margin: "12px 0" }}>
        <input type="file" accept="video/*" onChange={onFile} />
        <button onClick={ingest} disabled={busy}>
          {busy ? "Ingesting…" : file ? "Ingest video" : "Ingest demo (synthetic)"}
        </button>
        {status && (
          <span style={{ color: status.done ? "var(--hit)" : "var(--muted)" }}>
            {status.done ? `● ${status.frames} frames indexed` : "○ processing…"}
          </span>
        )}
      </div>

      <div style={{ position: "relative", background: "#000", borderRadius: 12, overflow: "hidden" }}>
        <video
          ref={videoRef}
          src={videoSrc || undefined}
          controls
          style={{ width: "100%", display: "block", maxHeight: 420 }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", inset: 0, pointerEvents: "none", width: "100%", height: "100%" }}
        />
      </div>

      {/* timeline */}
      <div style={{ position: "relative", height: 34, margin: "12px 0", background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8 }}>
        {results.map((r, i) => (
          <button
            key={i}
            title={`${(r.timestamp_ms / 1000).toFixed(1)}s · ${r.labels.join(", ")} · ${r.score.toFixed(2)}`}
            onClick={() => jumpTo(i)}
            style={{
              position: "absolute",
              left: `calc(${(r.timestamp_ms / durationMs) * 100}% - 6px)`,
              top: 6,
              width: 12,
              height: 22,
              padding: 0,
              borderRadius: 3,
              background: activeIdx === i ? "var(--hit)" : "var(--accent)",
            }}
          />
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, margin: "12px 0" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
          placeholder='e.g. "a car turning left"'
          style={{ flex: 1 }}
        />
        <button onClick={search} disabled={busy || !videoId}>
          Search
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {results.map((r, i) => (
          <div
            key={i}
            onClick={() => jumpTo(i)}
            style={{
              cursor: "pointer",
              padding: 10,
              borderRadius: 8,
              border: `1px solid ${activeIdx === i ? "var(--hit)" : "var(--border)"}`,
              background: "var(--panel)",
            }}
          >
            <strong>{(r.timestamp_ms / 1000).toFixed(1)}s</strong>{" "}
            <span style={{ color: "var(--muted)" }}>score {r.score.toFixed(3)}</span>
            <div style={{ color: "var(--muted)" }}>{r.labels.join(", ")}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16, fontFamily: "monospace", fontSize: 12, color: "var(--muted)" }}>
        {log.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
    </main>
  );
}
