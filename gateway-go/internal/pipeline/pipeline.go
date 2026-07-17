// Package pipeline provides the ingestion job queue, per-video status store, and
// the worker pool that spawns the Rust processor for each uploaded video.
//
// The queue is an in-process buffered channel here; swapping in Redis/Kafka means
// implementing Push + a consumer with the same shape. Go's cheap goroutines make
// the worker pool trivially concurrent.
package pipeline

import (
	"os/exec"
	"regexp"
	"strconv"
	"sync"

	"github.com/avase33/vidgrep/gateway/internal/config"
)

type Job struct {
	VideoID string
	Path    string // staged file path ("" for a synthetic job)
}

type Queue struct {
	ch chan Job
}

func NewQueue(buffer int) *Queue { return &Queue{ch: make(chan Job, buffer)} }

// Push enqueues without blocking; returns false if the queue is full.
func (q *Queue) Push(j Job) bool {
	select {
	case q.ch <- j:
		return true
	default:
		return false
	}
}

func (q *Queue) Chan() <-chan Job { return q.ch }

type Status struct {
	VideoID string `json:"video_id"`
	Frames  int    `json:"frames"`
	Done    bool   `json:"done"`
	Error   string `json:"error,omitempty"`
}

type Store struct {
	mu   sync.RWMutex
	m    map[string]*Status
	subs map[string][]chan Status
}

func NewStore() *Store {
	return &Store{m: make(map[string]*Status), subs: make(map[string][]chan Status)}
}

func (s *Store) Create(id string) {
	s.mu.Lock()
	s.m[id] = &Status{VideoID: id}
	s.mu.Unlock()
}

func (s *Store) Get(id string) (Status, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	st, ok := s.m[id]
	if !ok {
		return Status{}, false
	}
	return *st, true
}

func (s *Store) update(id string, fn func(*Status)) {
	s.mu.Lock()
	st, ok := s.m[id]
	if !ok {
		st = &Status{VideoID: id}
		s.m[id] = st
	}
	fn(st)
	snapshot := *st
	subs := append([]chan Status(nil), s.subs[id]...)
	s.mu.Unlock()
	for _, c := range subs {
		select {
		case c <- snapshot:
		default:
		}
	}
}

func (s *Store) Subscribe(id string) chan Status {
	c := make(chan Status, 8)
	s.mu.Lock()
	s.subs[id] = append(s.subs[id], c)
	s.mu.Unlock()
	return c
}

func (s *Store) Unsubscribe(id string, c chan Status) {
	s.mu.Lock()
	defer s.mu.Unlock()
	subs := s.subs[id]
	for i, sc := range subs {
		if sc == c {
			s.subs[id] = append(subs[:i], subs[i+1:]...)
			break
		}
	}
}

type Pool struct {
	cfg   config.Config
	queue *Queue
	store *Store
}

func NewPool(cfg config.Config, q *Queue, st *Store) *Pool {
	return &Pool{cfg: cfg, queue: q, store: st}
}

func (p *Pool) Run() {
	for i := 0; i < max(1, p.cfg.Workers); i++ {
		go p.loop()
	}
}

func (p *Pool) loop() {
	for job := range p.queue.Chan() {
		p.process(job)
	}
}

var sentRe = regexp.MustCompile(`sent (\d+)/(\d+)`)

func (p *Pool) process(job Job) {
	// Spawn the Rust processor; it decodes/samples frames and posts them to the
	// ML service for embedding + indexing.
	cmd := exec.Command(p.cfg.ProcessorBin, "--video-id", job.VideoID, "--ml-url", p.cfg.MLURL)
	out, err := cmd.CombinedOutput()
	if err != nil {
		p.store.update(job.VideoID, func(s *Status) {
			s.Done = true
			s.Error = "processor: " + err.Error()
		})
		return
	}
	frames := 0
	if m := sentRe.FindSubmatch(out); m != nil {
		frames, _ = strconv.Atoi(string(m[1]))
	}
	p.store.update(job.VideoID, func(s *Status) {
		s.Frames = frames
		s.Done = true
	})
}
