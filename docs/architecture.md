# Architecture

`mskamal1` is a small Go data/ML platform built around one idea from the
research in [go-data-ml-research.md](go-data-ml-research.md): **Go is the
productionisation layer** — ingestion, pipelines and inference serving —
while model research/training can stay in Python. The two sides meet at a
versioned artifact (`model.json` / `model.onnx`), never at a shared runtime.

## The vertical slice

```
            ┌─────────────┐   CSV    ┌────────────────┐  model.json  ┌─────────────┐
 data ─────▶│ cmd/ingest  │─────────▶│ cmd/pipeline   │─────────────▶│ cmd/serve   │──▶ HTTP
 (or the    │ Source →    │          │ split → train  │              │ Predictor   │    /v1/predict
 synthetic  │ Processors  │          │ → evaluate     │  model.onnx  │ (go | onnx) │
 generator) │ → Sink      │          └──────▲─────────┘──────────────└─────────────┘
            └─────────────┘                 │
                              scripts/train_export_onnx.py
                              (Python + numpy → same artifacts)
```

1. **ingest** (`internal/ingest`) — plugin-style `Source`/`Processor`/`Sink`
   interfaces (the Telegraf/Bento pattern). Shipped: CSV source/sink, JSONL
   sink, `DropIncomplete`, `Shuffle`, and a deterministic synthetic source so
   the demo needs no data download.
2. **pipeline** (`internal/pipeline`, `internal/features`, `internal/model`) —
   stages run in order with retry (`internal/pipeline.Runner`); a `Scaler` is
   fitted on the training split only; softmax regression is trained with
   deterministic full-batch gradient descent.
3. **serve** (`internal/server`, `internal/inference`) — stdlib HTTP API
   (`api/openapi.yaml`) over one `Predictor`. The fitted scaler travels inside
   the artifact, so train/serve skew is designed out.

## The two serving backends (and why both exist)

| | Pure Go (`backend: go`) | ONNX Runtime (`backend: onnx`) |
|---|---|---|
| Build | `make build` — CGO-free static binary | `make build-onnx` — CGO + `-tags onnx` |
| Loads | `model.json` | `model.onnx` (+ `model.json` for label order) |
| Use it for | tests, edge/embedded, the demo, anywhere CGO hurts | Python-trained models, HuggingFace exports, max throughput |
| Proved by | unit tests everywhere | `TestONNXParity` vs the pure-Go predictor (float32 tolerance) |

The ONNX code is isolated behind a build tag so the default binary and the
test suite never need the native library. Asking a non-ONNX binary for the
ONNX backend fails loudly (`ErrONNXNotBuilt`) — a serving binary must run the
backend its operator configured, never a silent substitute.

## Layout decisions

The layout follows the research (§3): start from `cmd/` + `internal/`, grow
only as needed, and **do not copy `golang-standards/project-layout` blindly** —
it is not an official standard and most of its folders would be empty here.

- `cmd/{ingest,pipeline,serve}` — one slim `main.go` per binary; they only
  parse flags/config and wire packages together.
- `internal/` — everything private; the compiler enforces that boundary.
  - `ingest` (plugin interfaces + CSV/synthetic), `features` (scaler),
    `model` (artifact schema, training, eval), `inference` (`Predictor`,
    Go + ONNX backends), `pipeline` (stage runner/retry),
    `server` (HTTP), `platform` (config/logging/metrics).
- `pkg/client` — the *only* public package, deliberately: a client for the
  serving API. If nothing external should import it, `pkg/` should not exist.
- `api/` OpenAPI contract; `configs/` JSON configs (stdlib only, no YAML
  dependency); `data/` git-ignored except the tiny synthetic sample;
  `models/` artifacts (`model.json` demo committed, `.onnx` regenerated —
  real weights belong in object storage/DVC/LFS, see research §3.3);
  `notebooks/` GoNB scratch, never production code; `deploy/`, `scripts/`.

## Data & model versioning

Git stores code and the artifact *schema*, not data. `data/**` is ignored
(generate the demo with `make run-ingest`), and `models/**/*.onnx` is ignored
(regenerate with `make train-onnx`). If datasets later need
branching/audit/rollback as a product feature, adopt Dolt; if files suffice,
DVC/Git LFS — see research §5.

## Concurrency note

`ONNXPredictor` owns one ONNX session whose tensors are bound at creation,
so `Predict` is mutex-serialised per predictor (documented in
`internal/inference/onnx.go`). The pure-Go predictor is stateless and fully
concurrent. Scale the ONNX backend with replicas or a predictor pool.

## Where to grow next

- **Streaming/ETL at scale:** adopt Bento's connector model (or Bento itself)
  behind the `ingest.Source` interface — one new file per connector.
- **Transformer/NLP features:** Hugot over ONNX rather than hand-rolled
  tokenisation (research §5).
- **Native Go training:** GoMLX behind `internal/model` if training must move
  into Go; pin versions (pre-1.0 API churn) and keep the artifact contract.
- **Metrics:** `internal/platform.Metrics` is intentionally minimal; swap in
  `prometheus/client_golang` behind the same handler when histograms arrive.
