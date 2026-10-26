// Package inference exposes one Predictor interface with two backends:
//
//   - ArtifactPredictor: pure Go, no CGO, loads model.json. Always available
//     and used for tests and the default binary (make build).
//   - ONNXPredictor: real ONNX Runtime via CGO (build tag "onnx", see
//     onnx.go), loading the model.onnx exported by
//     scripts/train_export_onnx.py (make build-onnx).
//
// Both must agree numerically; TestONNXParity (onnx_test.go) proves it.
package inference

import (
	"context"
	"errors"
	"fmt"

	"github.com/kamalbhue547/mskamal1/internal/model"
)

// ModelInfo describes a loaded model for the /v1/model endpoint.
type ModelInfo struct {
	Name         string   `json:"name"`
	Version      string   `json:"version"`
	Backend      string   `json:"backend"` // "go" or "onnx"
	FeatureNames []string `json:"feature_names"`
	Classes      []string `json:"classes"`
}

// Prediction is one inference result.
type Prediction struct {
	Label         string    `json:"label"`
	ClassIndex    int       `json:"class_index"`
	Probabilities []float64 `json:"probabilities"`
}

// Predictor serves predictions for one loaded model.
type Predictor interface {
	Predict(ctx context.Context, features []float64) (Prediction, error)
	Info() ModelInfo
	Close() error
}

// ArtifactPredictor is the pure-Go Predictor over a model.Artifact.
type ArtifactPredictor struct {
	artifact *model.Artifact
}

// NewArtifactPredictor validates and wraps an artifact.
func NewArtifactPredictor(a *model.Artifact) (*ArtifactPredictor, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	return &ArtifactPredictor{artifact: a}, nil
}

// LoadArtifactPredictor loads model.json from disk.
func LoadArtifactPredictor(path string) (*ArtifactPredictor, error) {
	a, err := model.LoadArtifact(path)
	if err != nil {
		return nil, err
	}
	return NewArtifactPredictor(a)
}

// Predict implements Predictor.
func (p *ArtifactPredictor) Predict(_ context.Context, features []float64) (Prediction, error) {
	idx, probs, err := p.artifact.Predict(features)
	if err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	return Prediction{Label: p.artifact.Classes[idx], ClassIndex: idx, Probabilities: probs}, nil
}

// Info implements Predictor.
func (p *ArtifactPredictor) Info() ModelInfo {
	return ModelInfo{
		Name:         p.artifact.Name,
		Version:      p.artifact.Version,
		Backend:      "go",
		FeatureNames: append([]string(nil), p.artifact.FeatureNames...),
		Classes:      append([]string(nil), p.artifact.Classes...),
	}
}

// Close implements Predictor (nothing to release).
func (p *ArtifactPredictor) Close() error { return nil }

// Artifact exposes the underlying artifact (used by the parity test).
func (p *ArtifactPredictor) Artifact() *model.Artifact { return p.artifact }

// ErrONNXNotBuilt is returned by the stub ONNX constructor in binaries built
// without the "onnx" build tag.
var ErrONNXNotBuilt = errors.New("binary built without ONNX support: rebuild with -tags onnx (CGO and the ONNX Runtime shared library are required)")

// Options selects and configures a Predictor backend.
type Options struct {
	// ModelPath is the model.json artifact (pure-Go backend / model metadata).
	ModelPath string
	// ONNXPath, when non-empty, selects the ONNX backend loading this file.
	ONNXPath string
	// ONNXLibPath is the full path to libonnxruntime.so (or .dylib/.dll).
	// Empty means: use ONNX_LIB_PATH, then the library loader's defaults.
	ONNXLibPath string
	// Classes and FeatureNames describe the ONNX graph when no model.json is
	// available alongside it. When ModelPath is set they come from there.
	Classes      []string
	FeatureNames []string
}

// NewPredictor returns the ONNX predictor when ONNXPath is set, otherwise the
// pure-Go artifact predictor. In a binary built without -tags onnx, setting
// ONNXPath returns ErrONNXNotBuilt — it never silently falls back, because a
// serving binary must run the backend its operator asked for.
func NewPredictor(opts Options) (Predictor, error) {
	if opts.ONNXPath != "" {
		return newONNXPredictor(opts)
	}
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("inference: one of ModelPath or ONNXPath is required")
	}
	return LoadArtifactPredictor(opts.ModelPath)
}
