package features

import (
	"math"
	"testing"

	"github.com/kamalbhue547/mskamal1/internal/ingest"
)

func testDataset() *ingest.Dataset {
	return &ingest.Dataset{
		FeatureNames: []string{"a", "b"},
		LabelName:    "label",
		Records: []ingest.Record{
			{Features: []float64{1, 10}, Label: "x"},
			{Features: []float64{3, 10}, Label: "y"},
		},
	}
}

func TestScalerFitTransform(t *testing.T) {
	var sc Scaler
	if err := sc.Fit(testDataset()); err != nil {
		t.Fatal(err)
	}
	if sc.Mean[0] != 2 || sc.Mean[1] != 10 {
		t.Fatalf("mean = %v", sc.Mean)
	}
	if math.Abs(sc.Scale[0]-1) > 1e-12 {
		t.Fatalf("scale[0] = %v, want 1", sc.Scale[0])
	}
	if sc.Scale[1] != 1 { // zero variance → guarded to 1
		t.Fatalf("scale[1] = %v, want guarded 1", sc.Scale[1])
	}
	out, err := sc.Transform([]float64{3, 12})
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != 1 || out[1] != 2 {
		t.Fatalf("transform = %v, want [1 2]", out)
	}
}

func TestScalerTransformLengthMismatch(t *testing.T) {
	var sc Scaler
	if err := sc.Fit(testDataset()); err != nil {
		t.Fatal(err)
	}
	if _, err := sc.Transform([]float64{1}); err == nil {
		t.Fatal("expected length mismatch error")
	}
}
