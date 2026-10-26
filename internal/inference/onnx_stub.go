//go:build !onnx

package inference

// newONNXPredictor is the no-CGO stub. Keeping the constructor behind the
// same signature lets cmd/serve wire backends identically in both builds.
func newONNXPredictor(Options) (Predictor, error) {
	return nil, ErrONNXNotBuilt
}
