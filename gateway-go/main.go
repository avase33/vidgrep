// Command vidgrep-gateway is the ingestion front door: it accepts large video
// uploads (chunk-streamed to disk), queues a processing job, runs a worker pool
// that spawns the Rust processor per video, and proxies search queries to the
// Python ML service.
package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/gorilla/websocket"

	"github.com/avase33/vidgrep/gateway/internal/config"
	"github.com/avase33/vidgrep/gateway/internal/pipeline"
)

type App struct {
	cfg   config.Config
	queue *pipeline.Queue
	store *pipeline.Store
}

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}

func newID() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return "v-" + hex.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// POST /upload  (multipart: video=@file)  — also accepts an empty body to enqueue
// a synthetic job for the offline demo.
func (a *App) handleUpload(w http.ResponseWriter, r *http.Request) {
	id := newID()
	path := ""

	if err := r.ParseMultipartForm(32 << 20); err == nil {
		if file, header, ferr := r.FormFile("video"); ferr == nil {
			defer file.Close()
			path = filepath.Join(a.cfg.TempDir, id+"_"+filepath.Base(header.Filename))
			dst, cerr := os.Create(path)
			if cerr != nil {
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": cerr.Error()})
				return
			}
			// stream the (potentially multi-GB) upload to disk in chunks
			if _, cerr = io.Copy(dst, file); cerr != nil {
				dst.Close()
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": cerr.Error()})
				return
			}
			dst.Close()
		}
	}

	a.store.Create(id)
	if !a.queue.Push(pipeline.Job{VideoID: id, Path: path}) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "queue full"})
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"video_id": id, "status": "queued"})
}

// GET /status/{id}
func (a *App) handleStatus(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	st, ok := a.store.Get(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown video_id"})
		return
	}
	writeJSON(w, http.StatusOK, st)
}

// POST /search  — proxy to the Python ML service.
func (a *App) handleSearch(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	resp, err := http.Post(a.cfg.MLURL+"/search", "application/json", bytes.NewReader(body))
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// GET /ws/status/{id}  — stream processing progress.
func (a *App) handleWSStatus(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	sub := a.store.Subscribe(id)
	defer a.store.Unsubscribe(id, sub)

	if st, ok := a.store.Get(id); ok {
		_ = conn.WriteJSON(st)
		if st.Done {
			return
		}
	}
	for {
		select {
		case st := <-sub:
			if err := conn.WriteJSON(st); err != nil {
				return
			}
			if st.Done {
				return
			}
		case <-time.After(60 * time.Second):
			return
		}
	}
}

func (a *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "service": "gateway", "ml": a.cfg.MLURL})
}

func main() {
	cfg := config.Load()
	app := &App{cfg: cfg, queue: pipeline.NewQueue(1024), store: pipeline.NewStore()}
	pipeline.NewPool(cfg, app.queue, app.store).Run()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /upload", app.handleUpload)
	mux.HandleFunc("GET /status/{id}", app.handleStatus)
	mux.HandleFunc("POST /search", app.handleSearch)
	mux.HandleFunc("GET /ws/status/{id}", app.handleWSStatus)
	mux.HandleFunc("GET /health", app.handleHealth)

	log.Printf("vidgrep gateway on %s (ml=%s, processor=%s, workers=%d)",
		cfg.Addr, cfg.MLURL, cfg.ProcessorBin, cfg.Workers)
	if err := http.ListenAndServe(cfg.Addr, mux); err != nil {
		log.Fatal(err)
	}
}
