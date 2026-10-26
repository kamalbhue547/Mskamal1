// Package platform holds cross-cutting concerns shared by every binary:
// configuration, structured logging and metrics. Nothing ML-specific lives here.
package platform

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// ServeConfig configures cmd/serve.
type ServeConfig struct {
	Addr               string `json:"addr"`
	ModelPath          string `json:"model_path"`
	ONNXPath           string `json:"onnx_path"`
	ONNXLibPath        string `json:"onnx_lib_path"`
	LogLevel           string `json:"log_level"`
	ReadTimeoutSeconds int    `json:"read_timeout_seconds"`
	WriteTimeoutSec    int    `json:"write_timeout_seconds"`
	ShutdownTimeoutSec int    `json:"shutdown_timeout_seconds"`
}

// PipelineConfig configures cmd/pipeline.
type PipelineConfig struct {
	InputPath    string  `json:"input_path"`
	ModelOutDir  string  `json:"model_out_dir"`
	ProcessedDir string  `json:"processed_dir"`
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
	L2           float64 `json:"l2"`
	TestFraction float64 `json:"test_fraction"`
	Seed         int64   `json:"seed"`
}

// DefaultServeConfig returns the serving defaults used when no file is given.
func DefaultServeConfig() ServeConfig {
	return ServeConfig{
		Addr:               ":8080",
		ModelPath:          "models/demo/model.json",
		LogLevel:           "info",
		ReadTimeoutSeconds: 10,
		WriteTimeoutSec:    10,
		ShutdownTimeoutSec: 10,
	}
}

// DefaultPipelineConfig returns the training defaults.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		InputPath:    "data/raw/sample.csv",
		ModelOutDir:  "models/demo",
		ProcessedDir: "data/processed",
		Epochs:       300,
		LearningRate: 0.1,
		L2:           0.001,
		TestFraction: 0.2,
		Seed:         42,
	}
}

// LoadServeConfig reads path (if non-empty) over the defaults, then applies
// SERVE_*/MSKAMAL1_* environment overrides. Flags applied by callers win last.
func LoadServeConfig(path string) (ServeConfig, error) {
	cfg := DefaultServeConfig()
	if path != "" {
		if err := loadJSON(path, &cfg); err != nil {
			return ServeConfig{}, err
		}
	}
	if v := os.Getenv("SERVE_ADDR"); v != "" {
		cfg.Addr = v
	}
	if v := os.Getenv("MODEL_PATH"); v != "" {
		cfg.ModelPath = v
	}
	if v := os.Getenv("ONNX_MODEL_PATH"); v != "" {
		cfg.ONNXPath = v
	}
	if v := os.Getenv("ONNX_LIB_PATH"); v != "" {
		cfg.ONNXLibPath = v
	}
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if cfg.Addr == "" {
		return ServeConfig{}, fmt.Errorf("serve addr must not be empty")
	}
	if cfg.ModelPath == "" && cfg.ONNXPath == "" {
		return ServeConfig{}, fmt.Errorf("one of model_path or onnx_path is required")
	}
	return cfg, nil
}

// LoadPipelineConfig reads path (if non-empty) over the defaults.
func LoadPipelineConfig(path string) (PipelineConfig, error) {
	cfg := DefaultPipelineConfig()
	if path != "" {
		if err := loadJSON(path, &cfg); err != nil {
			return PipelineConfig{}, err
		}
	}
	if cfg.Epochs <= 0 {
		return PipelineConfig{}, fmt.Errorf("epochs must be > 0, got %d", cfg.Epochs)
	}
	if cfg.LearningRate <= 0 {
		return PipelineConfig{}, fmt.Errorf("learning_rate must be > 0, got %v", cfg.LearningRate)
	}
	if cfg.TestFraction < 0 || cfg.TestFraction >= 1 {
		return PipelineConfig{}, fmt.Errorf("test_fraction must be in [0,1), got %v", cfg.TestFraction)
	}
	return cfg, nil
}

func loadJSON(path string, dst any) error {
	b, err := os.ReadFile(path) //nolint:gosec // path comes from operator config/flags
	if err != nil {
		return fmt.Errorf("read config %s: %w", path, err)
	}
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("parse config %s: %w", path, err)
	}
	return nil
}

// DurationSeconds converts a *_seconds config field to a time.Duration.
func DurationSeconds(s int) time.Duration { return time.Duration(s) * time.Second }

// EnvInt reads an int environment variable, falling back to def.
func EnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
