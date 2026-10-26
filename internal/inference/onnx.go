//go:build onnx

package inference

import (
	"context"
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/kamalbhue547/mskamal1/internal/model"
)

// ONNX graph contract, fixed by scripts/train_export_onnx.py:
//
//	input  "features":      float32 [1, n_features]  (raw, unscaled features)
//	output "probabilities": float32 [1, n_classes]   (softmax)
//	output "label":         int64   [1]              (argmax)
//
// The training-time standardisation is baked into the exported graph, so this
// predictor takes raw features exactly like ArtifactPredictor.

var (
	envOnce sync.Once
	envErr  error
)

func initONNXEnvironment(libPath string) error {
	envOnce.Do(func() {
		if libPath == "" {
			libPath = os.Getenv("ONNX_LIB_PATH")
		}
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		envErr = ort.InitializeEnvironment()
	})
	return envErr
}

// ONNXPredictor serves a model.onnx through ONNX Runtime (CGO).
//
// An AdvancedSession binds its input/output tensors at creation time, so a
// predictor owns one session and serialises Predict calls with a mutex:
// create one predictor per replica and scale horizontally, or pool several
// predictors if a single process needs concurrent ONNX throughput.
type ONNXPredictor struct {
	mu       sync.Mutex
	session  *ort.AdvancedSession
	input    *ort.Tensor[float32]
	probsOut *ort.Tensor[float32]
	labelOut *ort.Tensor[int64]
	info     ModelInfo
	closed   bool
}

func newONNXPredictor(opts Options) (Predictor, error) {
	classes, featureNames := opts.Classes, opts.FeatureNames
	name, version := "onnx-model", "1.0.0"
	// Prefer model.json metadata when available: labels must match the
	// training order exactly, and the artifact is the source of truth.
	if opts.ModelPath != "" {
		a, err := model.LoadArtifact(opts.ModelPath)
		if err != nil {
			return nil, err
		}
		classes, featureNames = a.Classes, a.FeatureNames
		name, version = a.Name, a.Version
	}
	if len(classes) == 0 || len(featureNames) == 0 {
		return nil, fmt.Errorf("onnx: classes and feature names are required (pass a model.json alongside the .onnx)")
	}
	if err := initONNXEnvironment(opts.ONNXLibPath); err != nil {
		return nil, fmt.Errorf("onnx: initialise environment: %w", err)
	}
	nFeat, nClass := int64(len(featureNames)), int64(len(classes))

	input, err := ort.NewTensor(ort.NewShape(1, nFeat), make([]float32, nFeat))
	if err != nil {
		return nil, fmt.Errorf("onnx: input tensor: %w", err)
	}
	probsOut, err := ort.NewEmptyTensor[float32](ort.NewShape(1, nClass))
	if err != nil {
		_ = input.Destroy()
		return nil, fmt.Errorf("onnx: probabilities tensor: %w", err)
	}
	labelOut, err := ort.NewEmptyTensor[int64](ort.NewShape(1))
	if err != nil {
		_ = input.Destroy()
		_ = probsOut.Destroy()
		return nil, fmt.Errorf("onnx: label tensor: %w", err)
	}
	session, err := ort.NewAdvancedSession(
		opts.ONNXPath,
		[]string{"features"},
		[]string{"probabilities", "label"},
		[]ort.Value{input},
		[]ort.Value{probsOut, labelOut},
		nil,
	)
	if err != nil {
		_ = input.Destroy()
		_ = probsOut.Destroy()
		_ = labelOut.Destroy()
		return nil, fmt.Errorf("onnx: create session for %s: %w", opts.ONNXPath, err)
	}
	return &ONNXPredictor{
		session:  session,
		input:    input,
		probsOut: probsOut,
		labelOut: labelOut,
		info: ModelInfo{
			Name:         name,
			Version:      version,
			Backend:      "onnx",
			FeatureNames: featureNames,
			Classes:      classes,
		},
	}, nil
}

// Predict implements Predictor.
func (p *ONNXPredictor) Predict(_ context.Context, features []float64) (Prediction, error) {
	if len(features) != len(p.info.FeatureNames) {
		return Prediction{}, fmt.Errorf("onnx: got %d features, want %d", len(features), len(p.info.FeatureNames))
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return Prediction{}, fmt.Errorf("onnx: predictor is closed")
	}
	data := p.input.GetData()
	for i, v := range features {
		data[i] = float32(v)
	}
	if err := p.session.Run(); err != nil {
		return Prediction{}, fmt.Errorf("onnx: run session: %w", err)
	}
	raw := p.probsOut.GetData()
	probs := make([]float64, len(raw))
	for i, v := range raw {
		probs[i] = float64(v)
	}
	idx := int(p.labelOut.GetData()[0])
	if idx < 0 || idx >= len(p.info.Classes) {
		return Prediction{}, fmt.Errorf("onnx: model returned out-of-range class index %d", idx)
	}
	return Prediction{Label: p.info.Classes[idx], ClassIndex: idx, Probabilities: probs}, nil
}

// Info implements Predictor.
func (p *ONNXPredictor) Info() ModelInfo { return p.info }

// Close releases the session and tensors. The ONNX environment itself is
// process-wide and intentionally left running.
func (p *ONNXPredictor) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	var err error
	if p.session != nil {
		err = p.session.Destroy()
	}
	for _, destroy := range []func() error{p.input.Destroy, p.probsOut.Destroy, p.labelOut.Destroy} {
		if e := destroy(); err == nil {
			err = e
		}
	}
	return err
}
