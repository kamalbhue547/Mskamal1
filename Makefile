GO ?= go
BIN_DIR := bin
PKG := ./...
# Set ONNX_LIB to the full path of libonnxruntime.so to run the ONNX parity
# tests / build the CGO serving binary, e.g.:
#   make onnx-test ONNX_LIB=$$(python3 -c 'import onnxruntime,os;print(os.path.join(os.path.dirname(onnxruntime.__file__),"capi","libonnxruntime.so"))')
ONNX_LIB ?=
ONNX_MODEL ?= models/demo/model.onnx

.PHONY: help fmt vet lint test test-race build build-onnx run-ingest run-pipeline run-serve train-onnx onnx-test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

fmt: ## Format code (gofumpt if available, else gofmt)
	@if command -v gofumpt >/dev/null 2>&1; then gofumpt -w .; else $(GO) fmt ./...; fi

vet: ## Run go vet
	$(GO) vet $(PKG)

lint: vet ## Lint (golangci-lint if installed, otherwise vet + staticcheck if present)
	@if command -v golangci-lint >/dev/null 2>&1; then golangci-lint run; \
	elif command -v staticcheck >/dev/null 2>&1; then staticcheck $(PKG); \
	else echo "golangci-lint/staticcheck not installed — ran go vet only"; fi

test: ## Run unit tests
	$(GO) test $(PKG)

test-race: ## Run tests with the race detector and coverage
	$(GO) test -race -coverprofile=coverage.out $(PKG)

build: ## Build all binaries into ./bin (pure Go, no CGO required)
	CGO_ENABLED=0 $(GO) build -o $(BIN_DIR)/ ./cmd/...

build-onnx: ## Build the serve binary with real ONNX Runtime support (CGO)
	CGO_ENABLED=1 $(GO) build -tags onnx -o $(BIN_DIR)/ ./cmd/...

run-ingest: ## Generate the synthetic sample dataset (data/raw/sample.csv)
	$(GO) run ./cmd/ingest -output data/raw/sample.csv

run-pipeline: build ## Train the demo model from data/raw/sample.csv
	./$(BIN_DIR)/pipeline -data data/raw/sample.csv -out models/demo

run-serve: build ## Serve the demo model on :8080
	./$(BIN_DIR)/serve -model models/demo/model.json

train-onnx: ## Train in Python (numpy) and export models/demo/model.{json,onnx}
	python3 scripts/train_export_onnx.py --data data/raw/sample.csv --out-dir models/demo

onnx-test: ## Parity-test the Go ONNX predictor against the pure-Go predictor
	@test -n "$(ONNX_LIB)" || (echo "set ONNX_LIB=/path/to/libonnxruntime.so" && exit 1)
	@test -f "$(ONNX_MODEL)" || (echo "$(ONNX_MODEL) missing — run 'make train-onnx' first" && exit 1)
	ONNX_LIB_PATH=$(ONNX_LIB) ONNX_MODEL_PATH=$(ONNX_MODEL) \
		CGO_ENABLED=1 $(GO) test -tags onnx ./internal/inference/ -run TestONNXParity -v

clean: ## Remove build artifacts
	rm -rf $(BIN_DIR) coverage.out coverage.html
