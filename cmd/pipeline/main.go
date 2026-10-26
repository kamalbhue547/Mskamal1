// Command pipeline trains the demo model:
//
//	load CSV → split train/test → fit scaler + train softmax regression →
//	evaluate → write models/<name>/model.json (+ processed splits, report)
//
// Training the production model in Python instead? scripts/train_export_onnx.py
// writes the same model.json schema plus model.onnx. The serving binary does
// not care which side produced the artifact.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/kamalbhue547/mskamal1/internal/ingest"
	"github.com/kamalbhue547/mskamal1/internal/model"
	"github.com/kamalbhue547/mskamal1/internal/pipeline"
	"github.com/kamalbhue547/mskamal1/internal/platform"
)

type job struct {
	cfg     platform.PipelineConfig
	dataset *ingest.Dataset
	trainDS *ingest.Dataset
	testDS  *ingest.Dataset
	art     *model.Artifact
	report  map[string]any
}

func main() {
	configPath := flag.String("config", "", "pipeline config JSON (default: configs/pipeline.json if present)")
	data := flag.String("data", "", "input CSV (overrides config)")
	out := flag.String("out", "", "model output dir (overrides config)")
	epochs := flag.Int("epochs", -1, "training epochs (overrides config)")
	lr := flag.Float64("lr", -1, "learning rate (overrides config)")
	flag.Parse()

	cfgFile := *configPath
	if cfgFile == "" {
		if _, err := os.Stat("configs/pipeline.json"); err == nil {
			cfgFile = "configs/pipeline.json"
		}
	}
	cfg, err := platform.LoadPipelineConfig(cfgFile)
	if err != nil {
		log.Fatalf("pipeline: %v", err)
	}
	if *data != "" {
		cfg.InputPath = *data
	}
	if *out != "" {
		cfg.ModelOutDir = *out
	}
	if *epochs > 0 {
		cfg.Epochs = *epochs
	}
	if *lr > 0 {
		cfg.LearningRate = *lr
	}

	logger := platform.NewLogger(os.Getenv("LOG_LEVEL"))
	j := &job{cfg: cfg}
	runner := pipeline.NewRunner(pipeline.Options{MaxAttempts: 1, Logger: logger})

	err = runner.Run(context.Background(),
		pipeline.StageFunc{StageName: "load", Fn: j.load},
		pipeline.StageFunc{StageName: "split", Fn: j.split},
		pipeline.StageFunc{StageName: "train", Fn: j.train},
		pipeline.StageFunc{StageName: "evaluate_and_save", Fn: j.evaluateAndSave},
	)
	if err != nil {
		log.Fatalf("pipeline: %v", err)
	}
	logger.Info("pipeline complete", "model_dir", cfg.ModelOutDir, "report", fmt.Sprint(j.report))
}

func (j *job) load(_ context.Context) error {
	ds, err := ingest.CSVSource{Path: j.cfg.InputPath}.Read(context.Background())
	if err != nil {
		return err
	}
	j.dataset = ds
	return nil
}

func (j *job) split(_ context.Context) error {
	j.trainDS, j.testDS = model.SplitTrainTest(j.dataset, j.cfg.TestFraction, j.cfg.Seed)
	if j.cfg.ProcessedDir != "" {
		if err := os.MkdirAll(j.cfg.ProcessedDir, 0o750); err != nil {
			return err
		}
		ctx := context.Background()
		if err := (ingest.CSVSink{Path: filepath.Join(j.cfg.ProcessedDir, "train.csv")}).Write(ctx, j.trainDS); err != nil {
			return err
		}
		if err := (ingest.CSVSink{Path: filepath.Join(j.cfg.ProcessedDir, "test.csv")}).Write(ctx, j.testDS); err != nil {
			return err
		}
	}
	return nil
}

func (j *job) train(_ context.Context) error {
	art, err := model.Train(j.trainDS, model.TrainOptions{
		Epochs:       j.cfg.Epochs,
		LearningRate: j.cfg.LearningRate,
		L2:           j.cfg.L2,
	})
	if err != nil {
		return err
	}
	j.art = art
	return nil
}

func (j *job) evaluateAndSave(_ context.Context) error {
	trainAcc, err := model.Accuracy(j.art, j.trainDS)
	if err != nil {
		return err
	}
	testAcc := trainAcc
	if len(j.testDS.Records) > 0 {
		if testAcc, err = model.Accuracy(j.art, j.testDS); err != nil {
			return err
		}
	}
	j.art.Metrics = map[string]float64{"train_accuracy": trainAcc, "test_accuracy": testAcc}

	if err := os.MkdirAll(j.cfg.ModelOutDir, 0o750); err != nil {
		return err
	}
	modelPath := filepath.Join(j.cfg.ModelOutDir, "model.json")
	if err := j.art.Save(modelPath); err != nil {
		return err
	}
	j.report = map[string]any{
		"model_path":     modelPath,
		"train_records":  len(j.trainDS.Records),
		"test_records":   len(j.testDS.Records),
		"train_accuracy": trainAcc,
		"test_accuracy":  testAcc,
		"classes":        j.art.Classes,
	}
	report, err := json.MarshalIndent(j.report, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(j.cfg.ModelOutDir, "report.json"), append(report, '\n'), 0o600)
}
