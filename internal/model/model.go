// Package model defines the portable model artifact and its pure-Go
// inference math.
//
// The artifact is deliberately boring JSON: the fitted feature scaler and a
// softmax-regression weight matrix. scripts/train_export_onnx.py writes
// exactly the same schema from Python and additionally exports model.onnx —
// where the scaler is baked into the first Gemm — so the pure-Go predictor
// (internal/inference) and the ONNX predictor agree numerically. That
// agreement is what the parity test in internal/inference verifies, mirroring
// the reference project in docs/go-data-ml-research.md §4.
package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/kamalbhue547/mskamal1/internal/features"
)

// Artifact is a trained model plus everything needed to serve it.
type Artifact struct {
	Name         string          `json:"name"`
	Version      string          `json:"version"`
	Task         string          `json:"task"` // "multiclass-classification"
	FeatureNames []string        `json:"feature_names"`
	Classes      []string        `json:"classes"`
	Scaler       features.Scaler `json:"scaler"`
	// Weights[feature][class], Bias[class]: logits = scaled(x)·W + b.
	Weights   [][]float64        `json:"weights"`
	Bias      []float64          `json:"bias"`
	TrainedAt time.Time          `json:"trained_at"`
	Metrics   map[string]float64 `json:"metrics,omitempty"`
}

// Validate checks the artifact is internally consistent and servable.
func (a *Artifact) Validate() error {
	if len(a.FeatureNames) == 0 {
		return fmt.Errorf("model: no feature names")
	}
	if len(a.Classes) < 2 {
		return fmt.Errorf("model: need at least 2 classes, got %d", len(a.Classes))
	}
	if len(a.Weights) != len(a.FeatureNames) {
		return fmt.Errorf("model: weights have %d rows, want %d features", len(a.Weights), len(a.FeatureNames))
	}
	for i, row := range a.Weights {
		if len(row) != len(a.Classes) {
			return fmt.Errorf("model: weights row %d has %d columns, want %d classes", i, len(row), len(a.Classes))
		}
	}
	if len(a.Bias) != len(a.Classes) {
		return fmt.Errorf("model: bias has %d entries, want %d classes", len(a.Bias), len(a.Classes))
	}
	if len(a.Scaler.Mean) != len(a.FeatureNames) || len(a.Scaler.Scale) != len(a.FeatureNames) {
		return fmt.Errorf("model: scaler dimension does not match feature count")
	}
	return nil
}

// Logits computes class logits for a raw (unscaled) feature vector.
func (a *Artifact) Logits(raw []float64) ([]float64, error) {
	scaled, err := a.Scaler.Transform(raw)
	if err != nil {
		return nil, err
	}
	logits := append([]float64(nil), a.Bias...)
	for f, v := range scaled {
		for c := range logits {
			logits[c] += v * a.Weights[f][c]
		}
	}
	return logits, nil
}

// Probabilities returns the softmax over Logits. Probabilities sum to 1.
func (a *Artifact) Probabilities(raw []float64) ([]float64, error) {
	logits, err := a.Logits(raw)
	if err != nil {
		return nil, err
	}
	return Softmax(logits), nil
}

// Predict returns the most likely class index and the full distribution.
func (a *Artifact) Predict(raw []float64) (classIdx int, probs []float64, err error) {
	probs, err = a.Probabilities(raw)
	if err != nil {
		return 0, nil, err
	}
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return best, probs, nil
}

// Softmax converts logits to probabilities (numerically stable).
func Softmax(logits []float64) []float64 {
	maxLogit := math.Inf(-1)
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - maxLogit)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Save writes the artifact as indented JSON.
func (a *Artifact) Save(path string) error {
	if err := a.Validate(); err != nil {
		return err
	}
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o600)
}

// LoadArtifact reads and validates an artifact written by Save (by Go or by
// scripts/train_export_onnx.py — the schema is shared).
func LoadArtifact(path string) (*Artifact, error) {
	b, err := os.ReadFile(path) //nolint:gosec // operator-supplied model path
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	var a Artifact
	if err := json.Unmarshal(b, &a); err != nil {
		return nil, fmt.Errorf("load model %s: %w", path, err)
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("load model %s: %w", path, err)
	}
	return &a, nil
}
