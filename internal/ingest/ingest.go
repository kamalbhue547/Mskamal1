// Package ingest defines the plugin-style Source → Processor → Sink contract
// for tabular ML data, borrowing the small-interface pattern that let
// Telegraf and Bento scale to hundreds of connectors (see
// docs/go-data-ml-research.md §4).
//
// A Dataset is the single unit flowing through every stage: named float
// features plus one string label per record. Add a new source or sink by
// implementing one interface — no pipeline code changes.
package ingest

import (
	"context"
	"fmt"
	"math"
	"math/rand"
)

// Record is one labelled example.
type Record struct {
	Features []float64 `json:"features"`
	Label    string    `json:"label"`
}

// Dataset is an in-memory tabular dataset.
type Dataset struct {
	FeatureNames []string `json:"feature_names"`
	LabelName    string   `json:"label_name"`
	Records      []Record `json:"records"`
}

// Validate checks shape consistency: every record must have len(FeatureNames)
// features and a non-empty label.
func (d *Dataset) Validate() error {
	if len(d.FeatureNames) == 0 {
		return fmt.Errorf("dataset has no feature names")
	}
	if len(d.Records) == 0 {
		return fmt.Errorf("dataset has no records")
	}
	for i, r := range d.Records {
		if len(r.Features) != len(d.FeatureNames) {
			return fmt.Errorf("record %d: got %d features, want %d", i, len(r.Features), len(d.FeatureNames))
		}
		if r.Label == "" {
			return fmt.Errorf("record %d: empty label", i)
		}
		for j, v := range r.Features {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("record %d feature %d: non-finite value %v", i, j, v)
			}
		}
	}
	return nil
}

// Classes returns the sorted-unique labels in first-seen order.
func (d *Dataset) Classes() []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range d.Records {
		if !seen[r.Label] {
			seen[r.Label] = true
			out = append(out, r.Label)
		}
	}
	return out
}

// Clone returns a deep copy (records and feature slices are not shared).
func (d *Dataset) Clone() *Dataset {
	out := &Dataset{
		FeatureNames: append([]string(nil), d.FeatureNames...),
		LabelName:    d.LabelName,
		Records:      make([]Record, len(d.Records)),
	}
	for i, r := range d.Records {
		out.Records[i] = Record{Features: append([]float64(nil), r.Features...), Label: r.Label}
	}
	return out
}

// Source produces a Dataset (a file, a stream, a generator, an API, …).
type Source interface {
	Name() string
	Read(ctx context.Context) (*Dataset, error)
}

// Processor transforms a Dataset in flight (cleaning, filtering, …).
type Processor interface {
	Name() string
	Process(ctx context.Context, ds *Dataset) (*Dataset, error)
}

// Sink persists a Dataset (CSV, JSONL, a database, object storage, …).
type Sink interface {
	Name() string
	Write(ctx context.Context, ds *Dataset) error
}

// ApplyProcessors runs processors in order, wrapping errors with the
// processor name so pipeline logs point at the failing stage.
func ApplyProcessors(ctx context.Context, ds *Dataset, procs ...Processor) (*Dataset, error) {
	var err error
	for _, p := range procs {
		ds, err = p.Process(ctx, ds)
		if err != nil {
			return nil, fmt.Errorf("processor %s: %w", p.Name(), err)
		}
	}
	return ds, nil
}

// DropIncomplete removes records containing NaN/Inf feature values.
type DropIncomplete struct{}

func (DropIncomplete) Name() string { return "drop_incomplete" }

// Process implements Processor.
func (DropIncomplete) Process(_ context.Context, ds *Dataset) (*Dataset, error) {
	out := ds.Clone()
	out.Records = out.Records[:0]
	for _, r := range ds.Records {
		ok := true
		for _, v := range r.Features {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				ok = false
				break
			}
		}
		if ok {
			out.Records = append(out.Records, Record{Features: append([]float64(nil), r.Features...), Label: r.Label})
		}
	}
	return out, nil
}

// Shuffle returns a processor that deterministically shuffles records.
func Shuffle(seed int64) Processor { return shuffleProcessor{seed: seed} }

type shuffleProcessor struct{ seed int64 }

func (s shuffleProcessor) Name() string { return "shuffle" }

func (s shuffleProcessor) Process(_ context.Context, ds *Dataset) (*Dataset, error) {
	out := ds.Clone()
	rnd := rand.New(rand.NewSource(s.seed)) //nolint:gosec // deterministic data shuffling, not security
	rnd.Shuffle(len(out.Records), func(i, j int) {
		out.Records[i], out.Records[j] = out.Records[j], out.Records[i]
	})
	return out, nil
}
