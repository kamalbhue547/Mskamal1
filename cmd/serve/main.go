// Command serve is the inference service: a slim wiring layer only —
// all logic lives in internal/{inference,server,platform}.
//
//	go run ./cmd/serve -config configs/serve.json
//	go run ./cmd/serve -model models/demo/model.json -addr :8080
//	# real ONNX Runtime backend (needs `make build-onnx`):
//	./bin/serve -onnx models/demo/model.onnx -onnx-lib /path/to/libonnxruntime.so
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/kamalbhue547/mskamal1/internal/inference"
	"github.com/kamalbhue547/mskamal1/internal/platform"
	"github.com/kamalbhue547/mskamal1/internal/server"
)

func main() {
	configPath := flag.String("config", "", "serve config JSON (default: configs/serve.json if present)")
	addr := flag.String("addr", "", "listen address (overrides config)")
	modelPath := flag.String("model", "", "model.json path (overrides config)")
	onnxPath := flag.String("onnx", "", "model.onnx path — selects the ONNX backend (requires -tags onnx build)")
	onnxLib := flag.String("onnx-lib", "", "path to libonnxruntime.so/.dylib/.dll")
	flag.Parse()

	cfgFile := *configPath
	if cfgFile == "" {
		if _, err := os.Stat("configs/serve.json"); err == nil {
			cfgFile = "configs/serve.json"
		}
	}
	cfg, err := platform.LoadServeConfig(cfgFile)
	if err != nil {
		log.Fatalf("serve: %v", err)
	}
	if *addr != "" {
		cfg.Addr = *addr
	}
	if *modelPath != "" {
		cfg.ModelPath = *modelPath
	}
	if *onnxPath != "" {
		cfg.ONNXPath = *onnxPath
	}
	if *onnxLib != "" {
		cfg.ONNXLibPath = *onnxLib
	}

	logger := platform.NewLogger(cfg.LogLevel)
	predictor, err := inference.NewPredictor(inference.Options{
		ModelPath:   cfg.ModelPath,
		ONNXPath:    cfg.ONNXPath,
		ONNXLibPath: cfg.ONNXLibPath,
	})
	if err != nil {
		logger.Error("load model failed", "error", err)
		os.Exit(1)
	}
	defer func() { _ = predictor.Close() }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := server.New(predictor, platform.NewMetrics(), logger)
	if err := srv.Serve(ctx,
		cfg.Addr,
		platform.DurationSeconds(cfg.ReadTimeoutSeconds),
		platform.DurationSeconds(cfg.WriteTimeoutSec),
		platform.DurationSeconds(cfg.ShutdownTimeoutSec),
	); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}
