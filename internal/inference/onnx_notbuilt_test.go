//go:build !onnx

package inference

import (
	"errors"
	"testing"
)

// In default (no-CGO) builds, asking for the ONNX backend must fail loudly —
// never silently serve a different backend than the operator configured.
func TestONNXBackendFailsLoudlyWithoutBuildTag(t *testing.T) {
	_, err := NewPredictor(Options{ONNXPath: "model.onnx"})
	if !errors.Is(err, ErrONNXNotBuilt) {
		t.Fatalf("err = %v, want ErrONNXNotBuilt", err)
	}
}
