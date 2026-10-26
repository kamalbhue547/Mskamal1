package inference

import (
	"context"
	"testing"

	"github.com/kamalbhue547/mskamal1/internal/ingest"
	"github.com/kamalbhue547/mskamal1/internal/model"
)

func demoPredictor(t *testing.T) Predictor {
	t.Helper()
	ds, err := ingest.SyntheticSource{Rows: 300, Seed: 42}.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	train, _ := model.SplitTrainTest(ds, 0.2, 42)
	a, err := model.Train(train, model.TrainOptions{Epochs: 300, LearningRate: 0.1, L2: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewArtifactPredictor(a)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestArtifactPredictor(t *testing.T) {
	p := demoPredictor(t)
	defer func() { _ = p.Close() }()

	pred, err := p.Predict(context.Background(), []float64{-2, -2, -2, -2})
	if err != nil {
		t.Fatal(err)
	}
	if pred.Label != "class_a" {
		t.Fatalf("label = %q, want class_a", pred.Label)
	}
	if p.Info().Backend != "go" || len(p.Info().Classes) != 3 {
		t.Fatalf("info = %+v", p.Info())
	}
	if _, err := p.Predict(context.Background(), []float64{1, 2}); err == nil {
		t.Fatal("expected feature-length error")
	}
}

func TestNewPredictorRequiresBackend(t *testing.T) {
	if _, err := NewPredictor(Options{}); err == nil {
		t.Fatal("expected error with no model configured")
	}
}
