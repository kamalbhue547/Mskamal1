package model

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/kamalbhue547/mskamal1/internal/features"
	"github.com/kamalbhue547/mskamal1/internal/ingest"
)

// TrainOptions controls softmax-regression training.
type TrainOptions struct {
	Epochs       int
	LearningRate float64
	L2           float64 // L2 regularisation strength (0 disables)
}

// SplitTrainTest deterministically shuffles ds and splits it by fraction.
// The returned datasets share the original feature names; records are copied.
func SplitTrainTest(ds *ingest.Dataset, testFraction float64, seed int64) (train, test *ingest.Dataset) {
	shuffled := ds.Clone()
	rnd := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic data split
	rnd.Shuffle(len(shuffled.Records), func(i, j int) {
		shuffled.Records[i], shuffled.Records[j] = shuffled.Records[j], shuffled.Records[i]
	})
	nTest := int(float64(len(shuffled.Records)) * testFraction)
	test = &ingest.Dataset{FeatureNames: ds.FeatureNames, LabelName: ds.LabelName, Records: shuffled.Records[:nTest]}
	train = &ingest.Dataset{FeatureNames: ds.FeatureNames, LabelName: ds.LabelName, Records: shuffled.Records[nTest:]}
	return train, test
}

// Train fits a scaler on train, then trains softmax regression with full-batch
// gradient descent on the scaled features. Training is deterministic for a
// given dataset and options, which keeps tests and demos reproducible.
func Train(train *ingest.Dataset, opts TrainOptions) (*Artifact, error) {
	if err := train.Validate(); err != nil {
		return nil, err
	}
	if opts.Epochs <= 0 || opts.LearningRate <= 0 {
		return nil, fmt.Errorf("train: epochs and learning rate must be > 0")
	}
	classes := train.Classes()
	classIdx := make(map[string]int, len(classes))
	for i, c := range classes {
		classIdx[c] = i
	}

	var scaler features.Scaler
	if err := scaler.Fit(train); err != nil {
		return nil, err
	}
	scaled, err := scaler.TransformDataset(train)
	if err != nil {
		return nil, err
	}

	nFeat, nClass := len(train.FeatureNames), len(classes)
	weights := make([][]float64, nFeat)
	for f := range weights {
		weights[f] = make([]float64, nClass)
	}
	bias := make([]float64, nClass)

	// Full-batch gradient descent on cross-entropy loss.
	for epoch := 0; epoch < opts.Epochs; epoch++ {
		gradW := make([][]float64, nFeat)
		for f := range gradW {
			gradW[f] = make([]float64, nClass)
		}
		gradB := make([]float64, nClass)
		for _, r := range scaled.Records {
			logits := append([]float64(nil), bias...)
			for f, v := range r.Features {
				for c := range logits {
					logits[c] += v * weights[f][c]
				}
			}
			probs := Softmax(logits)
			probs[classIdx[r.Label]]-- // dL/dlogits = p - onehot(y)
			for f, v := range r.Features {
				for c := range gradW[f] {
					gradW[f][c] += v * probs[c]
				}
			}
			for c := range gradB {
				gradB[c] += probs[c]
			}
		}
		n := float64(len(scaled.Records))
		for f := range weights {
			for c := range weights[f] {
				grad := gradW[f][c]/n + opts.L2*weights[f][c]
				weights[f][c] -= opts.LearningRate * grad
			}
		}
		for c := range bias {
			bias[c] -= opts.LearningRate * (gradB[c] / n)
		}
	}

	return &Artifact{
		Name:         "demo-softmax",
		Version:      "1.0.0",
		Task:         "multiclass-classification",
		FeatureNames: append([]string(nil), train.FeatureNames...),
		Classes:      classes,
		Scaler:       scaler,
		Weights:      weights,
		Bias:         bias,
		TrainedAt:    time.Now().UTC(),
	}, nil
}

// Accuracy evaluates a on ds and returns the fraction predicted correctly.
func Accuracy(a *Artifact, ds *ingest.Dataset) (float64, error) {
	if len(ds.Records) == 0 {
		return 0, fmt.Errorf("accuracy: empty dataset")
	}
	correct := 0
	for _, r := range ds.Records {
		idx, _, err := a.Predict(r.Features)
		if err != nil {
			return 0, err
		}
		if idx < len(a.Classes) && a.Classes[idx] == r.Label {
			correct++
		}
	}
	return float64(correct) / float64(len(ds.Records)), nil
}
