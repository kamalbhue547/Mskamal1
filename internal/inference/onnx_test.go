//go:build onnx

package inference

import (
	"context"
	"math"
	"os"
	"testing"
)

// resolveFromRepoRoot makes repo-root-relative artifact paths work even
// though `go test` runs with the package directory as working directory.
func resolveFromRepoRoot(p string) string {
	if _, err := os.Stat(p); err == nil {
		return p
	}
	if alt := "../../" + p; alt != p {
		if _, err := os.Stat(alt); err == nil {
			return alt
		}
	}
	return p
}

// TestONNXParity is the test this whole repository layout exists to make
// possible (mirroring go-onnx-production, docs/go-data-ml-research.md §4):
// a model trained/exported from Python must produce the same probabilities
// when served through ONNX Runtime from Go as through the pure-Go predictor
// over the same weights — within float32 tolerance.
//
// It is opt-in because it needs real artifacts and the native library:
//
//	make train-onnx   # writes models/demo/model.{json,onnx}
//	make onnx-test ONNX_LIB=/path/to/libonnxruntime.so
//
// Env: ONNX_MODEL_PATH (model.onnx), ONNX_LIB_PATH (libonnxruntime.so),
// MODEL_JSON_PATH (defaults to models/demo/model.json alongside the .onnx).
func TestONNXParity(t *testing.T) {
	onnxPath := os.Getenv("ONNX_MODEL_PATH")
	if onnxPath == "" {
		t.Skip("ONNX_MODEL_PATH not set — run `make onnx-test ONNX_LIB=...` (see Makefile)")
	}
	onnxPath = resolveFromRepoRoot(onnxPath)
	jsonPath := os.Getenv("MODEL_JSON_PATH")
	if jsonPath == "" {
		jsonPath = "models/demo/model.json"
	}
	jsonPath = resolveFromRepoRoot(jsonPath)

	goPred, err := LoadArtifactPredictor(jsonPath)
	if err != nil {
		t.Fatalf("load pure-Go predictor: %v", err)
	}
	onnxPred, err := NewPredictor(Options{ModelPath: jsonPath, ONNXPath: onnxPath, ONNXLibPath: os.Getenv("ONNX_LIB_PATH")})
	if err != nil {
		t.Fatalf("load ONNX predictor: %v", err)
	}
	defer func() { _ = onnxPred.Close() }()
	if onnxPred.Info().Backend != "onnx" {
		t.Fatalf("backend = %q, want onnx", onnxPred.Info().Backend)
	}

	cases := [][]float64{
		{-2, -2, -2, -2}, // class_a centre
		{0, 2, 0, 2},     // class_b centre
		{2, -1, 2, -1},   // class_c centre
		{-1.3, -2.4, -1.8, -2.2},
		{0.4, 1.5, -0.3, 2.4},
		{2.6, -0.7, 1.7, -1.5},
		{0, 0, 0, 0}, // between clusters: probabilities must still agree
	}
	const tol = 1e-5 // float32 graph vs float64 Go math
	for i, feats := range cases {
		want, err := goPred.Predict(context.Background(), feats)
		if err != nil {
			t.Fatalf("case %d: pure-Go predict: %v", i, err)
		}
		got, err := onnxPred.Predict(context.Background(), feats)
		if err != nil {
			t.Fatalf("case %d: ONNX predict: %v", i, err)
		}
		if got.Label != want.Label || got.ClassIndex != want.ClassIndex {
			t.Errorf("case %d: label = %q (idx %d), want %q (idx %d)", i, got.Label, got.ClassIndex, want.Label, want.ClassIndex)
		}
		for c := range want.Probabilities {
			if math.Abs(got.Probabilities[c]-want.Probabilities[c]) > tol {
				t.Errorf("case %d class %d: ONNX probability %v, pure-Go %v (tolerance %v)",
					i, c, got.Probabilities[c], want.Probabilities[c], tol)
			}
		}
	}
}
