// Package config loads gateway settings from the environment.
package config

import (
	"os"
	"strconv"
)

type Config struct {
	Addr         string // listen address
	MLURL        string // Python ML service base URL
	ProcessorBin string // path/name of the Rust processor binary to spawn
	TempDir      string // where uploaded videos are staged
	Workers      int    // size of the worker pool
}

func Load() Config {
	return Config{
		Addr:         env("VIDGREP_GATEWAY_ADDR", ":8080"),
		MLURL:        env("VIDGREP_ML_URL", "http://localhost:8000"),
		ProcessorBin: env("VIDGREP_PROCESSOR_BIN", "vidgrep-processor"),
		TempDir:      env("VIDGREP_TMP", os.TempDir()),
		Workers:      envInt("VIDGREP_WORKERS", 4),
	}
}

func env(k, def string) string {
	if v, ok := os.LookupEnv(k); ok && v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v, ok := os.LookupEnv(k); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
