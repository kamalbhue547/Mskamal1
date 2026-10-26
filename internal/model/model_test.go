package model

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/kamalbhue547/mskamal1/internal/ingest"
)

func trainDemo(t *testing.T) (*Artifact, *ingest.Dataset) {
	t.Helper()
	ds, err := ingest.SyntheticSource{Rows: 300, Seed: 42}.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	train, _ := SplitTrainTest(ds, 0.2, 42)
	a, err := Train(train, TrainOptions{Epochs: 300, LearningRate: 0.1, L2: 0.001, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	return a, ds
}

func TestTrainAchievesHighAccuracyOnSyntheticData(t *testing.T) {
	a, ds := trainDemo(t)
	acc, err := Accuracy(a, ds)
	if err != nil {
		t.Fatal(err)
	}
	if acc < 0.90 {
		t.Fatalf("accuracy = %.3f, want >= 0.90 on well-separated synthetic data", acc)
	}
}

func TestPredictProbabilitiesSumToOne(t *testing.T) {
	a, _ := trainDemo(t)
	idx, probs, err := a.Predict([]float64{-2, -2, -2, -2})
	if err != nil {
		t.Fatal(err)
	}
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("probabilities sum to %v", sum)
	}
	if a.Classes[idx] != "class_a" {
		t.Fatalf("predicted %q for the class_a centre, want class_a", a.Classes[idx])
	}
}

func TestArtifactSaveLoadRoundTrip(t *testing.T) {
	a, _ := trainDemo(t)
	path := filepath.Join(t.TempDir(), "model.json")
	if err := a.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := LoadArtifact(path)
	if err != nil {
		t.Fatal(err)
	}
	if back.Name != a.Name || len(back.Classes) != len(a.Classes) || back.Weights[0][0] != a.Weights[0][0] {
		t.Fatalf("round trip mismatch: %+v", back)
	}
	if _, err := LoadArtifact(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("expected error for missing file")
	}
}
