# Mskamal1

Software Developer | Building cool things for the web | Open Source Enthusiast

---

# mskamal1 — a Go data/ML platform

A working data/ML platform in Go, built from the research in
[`docs/go-data-ml-research.md`](docs/go-data-ml-research.md): ingest →
pipeline → train → serve, with **two interchangeable serving backends** —
pure Go (no CGO) and real ONNX Runtime (CGO) — that are parity-tested against
each other.

> The pattern this repo implements (research §1): **train in Python → export
> ONNX → serve in Go.** A pure-Go training path produces the identical
> artifact schema, so the demo runs anywhere with zero setup.

## Quickstart

Needs Go 1.23+ (CI uses 1.26). No third-party Go dependencies for the default
build; Python is only needed for the ONNX export path.

```bash
make run-ingest     # deterministic synthetic dataset -> data/raw/sample.csv
make run-pipeline   # train -> models/demo/model.json   (test accuracy ~1.0)
make run-serve      # serve on :8080

curl localhost:8080/v1/model
curl -X POST localhost:8080/v1/predict \
  -H 'Content-Type: application/json' -d '{"features":[-2,-2,-2,-2]}'
# {"label":"class_a","class_index":0,"probabilities":[0.99,...],
#  "model":"demo-softmax","backend":"go"}
curl localhost:8080/metrics   # Prometheus text
```

Or one-shot without the Makefile: `go run ./cmd/ingest`,
`go run ./cmd/pipeline`, `go run ./cmd/serve -config configs/serve.json`.

## Real ONNX serving

```bash
pip install numpy onnx onnxruntime
make train-onnx                                  # Python -> model.{json,onnx}
scripts/download_onnxruntime.sh                  # or reuse the wheel's library
export ONNX_LIB_PATH="$PWD/third_party/onnxruntime/lib/libonnxruntime.so"
make onnx-test ONNX_LIB="$ONNX_LIB_PATH"         # Go vs ONNX parity (<=1e-5)
make build-onnx
./bin/serve -model models/demo/model.json -onnx models/demo/model.onnx \
            -onnx-lib "$ONNX_LIB_PATH"
```

Full walkthrough: [`docs/onnx-serving.md`](docs/onnx-serving.md).

## Layout

```text
cmd/            ingest / pipeline / serve — slim main.go wiring only
internal/
  ingest/       Source → Processor → Sink plugin interfaces (CSV, JSONL, synthetic)
  features/     fitted transforms (scaler travels inside the artifact)
  model/        artifact schema + softmax-regression training/eval
  inference/    Predictor interface: pure-Go backend + ONNX backend (-tags onnx)
  pipeline/     stage runner with retry
  server/       stdlib HTTP API (see api/openapi.yaml)
  platform/     config, structured logging, Prometheus metrics
pkg/client/     the only public package: a client for the serving API
api/            OpenAPI contract        configs/  JSON configs (no YAML dep)
data/           raw/interim/processed — git-ignored, sample.csv excepted
models/         model.json committed (tiny demo), model.onnx regenerated
notebooks/      GoNB EDA scratch — never production code
deploy/         multi-stage Dockerfile + compose
scripts/        Python ONNX exporter, ONNX Runtime downloader
docs/           architecture, ONNX serving guide, source research
```

Why this shape — and why *not* `golang-standards/project-layout` — is
explained in [`docs/architecture.md`](docs/architecture.md).

## API

| Endpoint | Description |
|---|---|
| `GET /healthz`, `GET /readyz` | probes |
| `GET /v1/model` | loaded model, backend (`go`/`onnx`), features, classes |
| `POST /v1/predict` | `{"features":[...]}` (raw, unscaled) → label + probabilities |
| `GET /metrics` | Prometheus text (`predictions_total`, `http_requests_total`, …) |

Contract: [`api/openapi.yaml`](api/openapi.yaml). Go client: `pkg/client`.

## Development

```bash
make fmt        # gofumpt (falls back to gofmt)
make lint       # golangci-lint if present, else vet/staticcheck
make test       # unit tests — no CGO, no network, no native libs
make test-race  # race detector + coverage
make build      # bin/{ingest,pipeline,serve}, pure Go
```

CI (`.github/workflows/ci.yml`) runs gofumpt + vet + `go test -race`, a
lint job, a serving smoke test, and a dedicated **Python → ONNX → Go parity**
job — the workflow this repository exists to prove.

## Design choices worth knowing

- **Stdlib-first.** The default build has zero third-party Go dependencies.
  The single exception, `yalue/onnxruntime_go`, is isolated behind the `onnx`
  build tag so it never leaks into the default binary or test suite.
- **The scaler lives in the artifact.** Train/serve skew is designed out:
  both backends take raw features, identically (`docs/onnx-serving.md`).
- **No silent fallbacks.** A binary built without `-tags onnx` that is asked
  for the ONNX backend fails at startup, loudly.
- **Data and weights are not Git blobs.** Datasets and `.onnx` files are
  git-ignored and regenerated; see `docs/architecture.md` for the
  Dolt/DVC/object-storage decision path.

## Roadmap

- [ ] Second connector behind `ingest.Source` (HTTP/Kafka) — Bento pattern
- [ ] Predictor pool for concurrent ONNX serving
- [ ] GoMLX-backed native training behind `internal/model` (research §5)
- [ ] Hugot transformer pipeline as a second `Predictor`
- [ ] Dolt-backed dataset versioning

## License

MIT — see [LICENSE](LICENSE).
