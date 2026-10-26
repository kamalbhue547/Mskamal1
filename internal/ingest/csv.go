package ingest

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

// CSVSource reads a CSV whose header is feature names followed by the label
// column, e.g.:
//
//	feature_1,feature_2,feature_3,feature_4,label
//	-2.1,-1.8,-2.4,-1.9,class_a
type CSVSource struct{ Path string }

func (s CSVSource) Name() string { return "csv_source" }

// Read implements Source.
func (s CSVSource) Read(_ context.Context) (*Dataset, error) {
	f, err := os.Open(s.Path) //nolint:gosec // operator-supplied data path
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer func() { _ = f.Close() }()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse csv %s: %w", s.Path, err)
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("csv %s: need a header and at least one record", s.Path)
	}
	header := rows[0]
	if len(header) < 2 {
		return nil, fmt.Errorf("csv %s: header must have >=1 feature column plus a label column", s.Path)
	}
	ds := &Dataset{FeatureNames: header[:len(header)-1], LabelName: header[len(header)-1]}
	for i, row := range rows[1:] {
		if len(row) != len(header) {
			return nil, fmt.Errorf("csv %s line %d: got %d columns, want %d", s.Path, i+2, len(row), len(header))
		}
		feats := make([]float64, len(header)-1)
		for j := range feats {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, fmt.Errorf("csv %s line %d column %q: %w", s.Path, i+2, header[j], err)
			}
			feats[j] = v
		}
		ds.Records = append(ds.Records, Record{Features: feats, Label: row[len(row)-1]})
	}
	return ds, ds.Validate()
}

// CSVSink writes a Dataset in the same format CSVSource reads.
type CSVSink struct{ Path string }

func (s CSVSink) Name() string { return "csv_sink" }

// Write implements Sink.
func (s CSVSink) Write(_ context.Context, ds *Dataset) error {
	f, err := os.Create(s.Path) //nolint:gosec // operator-supplied data path
	if err != nil {
		return fmt.Errorf("create csv: %w", err)
	}
	defer func() { _ = f.Close() }()
	w := csv.NewWriter(f)
	header := append(append([]string(nil), ds.FeatureNames...), ds.LabelName)
	if err := w.Write(header); err != nil {
		return err
	}
	for _, r := range ds.Records {
		row := make([]string, 0, len(r.Features)+1)
		for _, v := range r.Features {
			row = append(row, strconv.FormatFloat(v, 'g', -1, 64))
		}
		if err := w.Write(append(row, r.Label)); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// JSONLSink writes one JSON Record per line (handy for streaming tools).
type JSONLSink struct{ Path string }

func (s JSONLSink) Name() string { return "jsonl_sink" }

// Write implements Sink.
func (s JSONLSink) Write(_ context.Context, ds *Dataset) error {
	f, err := os.Create(s.Path) //nolint:gosec // operator-supplied data path
	if err != nil {
		return fmt.Errorf("create jsonl: %w", err)
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)
	for _, r := range ds.Records {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}

// SyntheticSource generates a deterministic, well-separated 3-class dataset.
// It exists so the demo pipeline runs with no external data download; replace
// it with CSVSource (or your own Source) for real data.
type SyntheticSource struct {
	Rows int
	Seed int64
}

func (s SyntheticSource) Name() string { return "synthetic_source" }

// Read implements Source.
func (s SyntheticSource) Read(_ context.Context) (*Dataset, error) {
	rows := s.Rows
	if rows <= 0 {
		rows = 300
	}
	centers := map[string][]float64{
		"class_a": {-2, -2, -2, -2},
		"class_b": {0, 2, 0, 2},
		"class_c": {2, -1, 2, -1},
	}
	labels := []string{"class_a", "class_b", "class_c"}
	rnd := rand.New(rand.NewSource(s.Seed)) //nolint:gosec // synthetic demo data
	ds := &Dataset{
		FeatureNames: []string{"feature_1", "feature_2", "feature_3", "feature_4"},
		LabelName:    "label",
	}
	for i := 0; i < rows; i++ {
		label := labels[i%len(labels)]
		c := centers[label]
		feats := make([]float64, len(c))
		for j := range c {
			feats[j] = math.Round((c[j]+0.7*rnd.NormFloat64())*1000) / 1000
		}
		ds.Records = append(ds.Records, Record{Features: feats, Label: label})
	}
	rnd.Shuffle(len(ds.Records), func(i, j int) {
		ds.Records[i], ds.Records[j] = ds.Records[j], ds.Records[i]
	})
	return ds, ds.Validate()
}
