// Package features holds feature-engineering transforms. Transforms are
// fitted on training data only, persisted inside the model artifact, and
// re-applied identically at serving time — the classic train/serve skew trap
// is designed out by keeping the fitted state with the weights.
package features

import (
	"fmt"
	"math"

	"github.com/kamalbhue547/mskamal1/internal/ingest"
)

// Scaler standardises features to zero mean / unit variance.
type Scaler struct {
	Mean  []float64 `json:"mean"`
	Scale []float64 `json:"scale"`
}

// Fit computes per-feature mean and population standard deviation.
// Zero-variance features get scale 1 so division is always safe.
func (s *Scaler) Fit(ds *ingest.Dataset) error {
	if err := ds.Validate(); err != nil {
		return err
	}
	n := len(ds.FeatureNames)
	s.Mean = make([]float64, n)
	for _, r := range ds.Records {
		for j, v := range r.Features {
			s.Mean[j] += v
		}
	}
	for j := range s.Mean {
		s.Mean[j] /= float64(len(ds.Records))
	}
	s.Scale = make([]float64, n)
	for _, r := range ds.Records {
		for j, v := range r.Features {
			d := v - s.Mean[j]
			s.Scale[j] += d * d
		}
	}
	for j := range s.Scale {
		s.Scale[j] = math.Sqrt(s.Scale[j] / float64(len(ds.Records)))
		if s.Scale[j] == 0 {
			s.Scale[j] = 1
		}
	}
	return nil
}

// Transform scales one feature vector. It never mutates its input.
func (s *Scaler) Transform(features []float64) ([]float64, error) {
	if len(features) != len(s.Mean) {
		return nil, fmt.Errorf("scaler: got %d features, want %d", len(features), len(s.Mean))
	}
	out := make([]float64, len(features))
	for j, v := range features {
		out[j] = (v - s.Mean[j]) / s.Scale[j]
	}
	return out, nil
}

// TransformDataset scales every record in a copy of ds.
func (s *Scaler) TransformDataset(ds *ingest.Dataset) (*ingest.Dataset, error) {
	out := ds.Clone()
	for i := range out.Records {
		scaled, err := s.Transform(out.Records[i].Features)
		if err != nil {
			return nil, err
		}
		out.Records[i].Features = scaled
	}
	return out, nil
}
