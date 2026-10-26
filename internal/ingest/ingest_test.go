package ingest

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSyntheticSourceIsDeterministicAndValid(t *testing.T) {
	ctx := context.Background()
	a, err := SyntheticSource{Rows: 90, Seed: 42}.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	b, err := SyntheticSource{Rows: 90, Seed: 42}.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Records) != 90 || len(a.Classes()) != 3 {
		t.Fatalf("records=%d classes=%d", len(a.Records), len(a.Classes()))
	}
	for i := range a.Records {
		if a.Records[i].Label != b.Records[i].Label || a.Records[i].Features[0] != b.Records[i].Features[0] {
			t.Fatalf("record %d differs between runs with the same seed", i)
		}
	}
}

func TestCSVRoundTrip(t *testing.T) {
	ctx := context.Background()
	ds, err := SyntheticSource{Rows: 30, Seed: 7}.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "data.csv")
	if err := (CSVSink{Path: path}).Write(ctx, ds); err != nil {
		t.Fatal(err)
	}
	back, err := CSVSource{Path: path}.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Records) != len(ds.Records) || back.LabelName != "label" {
		t.Fatalf("round trip mismatch: %+v", back.FeatureNames)
	}
	for i := range ds.Records {
		if back.Records[i].Label != ds.Records[i].Label {
			t.Fatalf("record %d label mismatch", i)
		}
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestDropIncompleteProcessor(t *testing.T) {
	ds := &Dataset{
		FeatureNames: []string{"x"},
		LabelName:    "label",
		Records: []Record{
			{Features: []float64{1}, Label: "a"},
			{Features: []float64{math.NaN()}, Label: "a"},
		},
	}
	out, err := ApplyProcessors(context.Background(), ds, DropIncomplete{})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Records) != 1 {
		t.Fatalf("got %d records, want 1", len(out.Records))
	}
}
