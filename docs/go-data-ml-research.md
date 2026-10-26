# Go for Data / ML — Best Repository Research

*Prepared 26 Oct 2026. Scope: data/ML project in Go — starter templates + exemplary repositories to study/fork.*

---

## 1. TL;DR — the headline findings

1. **Go's sweet spot is data engineering and inference serving, not model research.** Single static binary, low memory, goroutine concurrency, and fast startup make Go excellent for pipelines and high-throughput prediction services; the exploratory/notebook + GPU-training ecosystem still belongs to Python. The dominant production pattern is: **train in Python (PyTorch) → export ONNX → serve in Go** ([Calmops architecture guide](https://calmops.com/programming/golang/golang-in-ai-applications-and-libraries/), [Go FAQ](https://www.gofaq.org/en/how-to-use-tensorflow-or-onnx-runtime-from-go/)).
2. **Do not blindly copy `golang-standards/project-layout`.** Despite the name it is *not* an official Go standard — Russ Cox and the core team have explicitly disavowed it, and its own README says it is overkill for simple projects. Start flat (`main.go` + `go.mod`) and grow into `cmd/` + `internal/` only when the project earns it ([levelupgo 2026 guide](https://levelupgo.dev/blog/go-project-structure), [project-layout README](https://github.com/golang-standards/project-layout/blob/master/README.md)).
3. **The Go ML torch has been passed.** Gorgonia (the historic deep-learning library, ~5.9k stars) is now lightly maintained; **GoMLX is the modern, actively developed choice** (Apache-2.0, OpenXLA backend, pure-Go backend, ONNX support, releases through 2026) ([libhunt activity comparison](https://www.libhunt.com/compare-gorgonia-vs-goml), [GoMLX GitHub](https://github.com/gomlx/gomlx)).
4. **Best all-around repository to study: `warpstreamlabs/bento`** (MIT-licensed fork of Benthos) — a battle-tested, plugin-driven data-streaming codebase that demonstrates exactly how a production Go data platform should be built. For ML-specific architecture, study **GoMLX** and the **ONNX Runtime Go** bindings.
5. **Licensing landmine to know:** Benthos was acquired by Redpanda in May 2024 and split into Redpanda Connect (connectors mostly Apache-2.0, two enterprise-gated) and a MIT core engine; WarpStream forked the last fully-MIT version as **Bento** ([WarpStream announcement](https://www.warpstream.com/blog/announcing-bento-the-open-source-fork-of-the-project-formerly-known-as-benthos), [Redpanda blog](https://www.redpanda.com/blog/redpanda-connect)).

---

## 2. The 2026 Go data/ML landscape (what's current)

| Layer | Recommended choice | State of play |
|---|---|---|
| Numerics (the "NumPy") | **[gonum/gonum](https://github.com/gonum/gonum)** | Mature, well-maintained foundation: matrices, BLAS/LAPACK, stats, optimization, `gonum/plot` |
| Native ML framework | **[gomlx/gomlx](https://github.com/gomlx/gomlx)** | Active in 2026 (v0.28.x, ~1.6k stars, Apache-2.0, Go 1.25+). XLA JIT to CPU/GPU/TPU, pure-Go portable backend (WASM-capable), ONNX import/export, reads HuggingFace models |
| Classic deep learning | [gorgonia/gorgonia](https://github.com/gorgonia/gorgonia) | Functional but no longer actively developed ("not as actively developed as it once was, it remains functional" — [Medium, Nov 2025](https://medium.com/@vladimirvivien/go-in-the-ai-ml-landscape-a-practical-guide-d36d44f360d2)); use GoMLX for new work |
| Classical ML (kNN, trees…) | [sjwhitworth/golearn](https://github.com/sjwhitworth/golearn) | "Batteries-included" scikit-learn analogue; modest maintenance |
| Inference runtime | **[yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go)** and the new official **[microsoft/onnxruntime/go](https://github.com/microsoft/onnxruntime/go)** | Community wrapper is mature & cross-platform; Microsoft's official Go bindings appeared in 2026 (used against ORT 1.29–1.31) — expect this to become the default |
| Transformer pipelines | **[knights-analytics/hugot](https://github.com/knights-analytics/hugot)** | HuggingFace-style text/vision/audio pipelines in Go over ONNX, with pure-Go / ORT / XLA backends |
| Dataframes | [go-gota/gota](https://github.com/go-gota/gota) | ⚠️ Largely unmaintained ("last updated 3 years ago", no vectorization/parallelism — [Reddit r/golang](https://www.reddit.com/r/golang/comments/1e7hw4j/is_there_any_chance_of_using_golang_for_ai_apps_in_2024_and_beyond/)). Evaluate `dataframe-go` instead; Gonum covers most numeric needs |
| Notebooks (EDA) | **[janpfeifer/gonb](https://github.com/janpfeifer/gonb)** | Full Jupyter kernel using the real Go compiler (CGO-compatible), actively released (v0.11.5, Jul 2026); ships inside the GoMLX JupyterLab Docker image |
| Data versioning | **[dolthub/dolt](https://github.com/dolthub/dolt)** | "Git for data" — version-controlled MySQL-compatible DB, 24.7k stars, Apache-2.0, very active; v2.0 (2026) adds GC, compression, vector index beta ([InfoQ](https://www.infoq.com/news/2026/07/dolt-version-control/)) |

Curated directories worth bookmarking: [promacanthus/awesome-golang-ai](https://github.com/promacanthus/awesome-golang-ai) and the older but thorough [gopherdata/resources](https://github.com/gopherdata/resources/blob/master/tooling/README.md).

---

## 3. How to structure the repository

### 3.1 What the official guidance actually says

There is no mandated Go directory tree. The official *Organizing a Go module* guide scales layout with project size, and the 2026 consensus is simple ([levelupgo](https://levelupgo.dev/blog/go-project-structure), [daveamit.com](https://daveamit.com/posts/2026-02-13-go-folder-structure/)):

- **Small tool / PoC** → flat: `main.go` + `go.mod` at root.
- **One binary with private packages** → add `internal/` (the *only* boundary the compiler enforces — external modules cannot import it).
- **Two+ binaries** → add `cmd/<binary>/main.go` each, keeping `main()` under ~50 lines.
- **Never add empty folders "for the future"**, never use `src/`, and treat `pkg/` as optional (the standard library doesn't use it).

### 3.2 Tooling to bootstrap

- **[`gonew`](https://go.dev/blog/gonew)** — the Go team's official (experimental) template instantiator: `go install golang.org/x/tools/cmd/gonew@latest && gonew <template-module> <your.module/path>`.
- **[link-foundation/go-ai-driven-development-pipeline-template](https://github.com/link-foundation/go-ai-driven-development-pipeline-template)** — the closest ready-made "production Go" template found: Go 1.21+, GitHub Actions CI across Ubuntu/macOS/Windows, race detection + coverage, `gofmt`/`go vet`/`staticcheck`, pre-commit hooks, changeset-based releases. Use it for the *engineering scaffolding*, then add the data/ML directories below.

### 3.3 Recommended layout for a Go data/ML project

A synthesis of idiomatic Go layout (`cmd`/`internal`, compiler-enforced privacy) with the classic ML project separation (raw/interim/processed data, model artifacts, configs, notebooks):

```text
my-data-ml-platform/
├── cmd/                      # one slim main.go per binary
│   ├── ingest/               # batch/stream ingestion job
│   ├── pipeline/             # feature-build / training-run orchestrator
│   └── serve/                # HTTP/gRPC inference service
├── internal/                 # private, compiler-enforced
│   ├── ingest/               # source connectors & readers
│   ├── features/             # feature engineering, transformations
│   ├── model/                # training logic / model wrappers (GoMLX etc.)
│   ├── inference/            # ONNX session mgmt, pre/post-processing
│   ├── pipeline/             # stage DAG, retry/checkpoint semantics
│   └── platform/             # config, logging, metrics, storage clients
├── pkg/                      # ONLY code deliberately exposed to importers
├── api/                      # OpenAPI / protobuf contracts
├── configs/                  # YAML/TOML: pipeline & hyperparameter configs
├── data/
│   ├── raw/                  # immutable originals (git-ignored)
│   ├── interim/              # transformed intermediates (git-ignored)
│   └── processed/            # modeling-ready datasets (git-ignored)
├── models/                   # exported ONNX artifacts — DVC/LFS/object store, not Git blobs
├── notebooks/                # GoNB EDA notebooks (experimentation only)
├── deploy/                   # multi-stage Dockerfile, compose, k8s manifests
├── scripts/  test/  docs/
├── .github/workflows/        # ci.yml: gofumpt + golangci-lint, go vet, staticcheck, go test -race, coverage
├── Makefile (or Taskfile.yml)
├── go.mod  go.sum  README.md  LICENSE  .gitignore
```

Key conventions, borrowed from the best reference repos:

- **`main.go` files only wire things together** — all real logic in `internal/` (the pattern [documented in project-layout best practices](https://deepwiki.com/golang-standards/project-layout/8-best-practices)).
- **Never commit data or model weights to Git.** Git-ignore `data/` and `models/`; track artifacts with DVC/Git LFS/S3, or version the *data itself* with Dolt.
- **CI tooling match the leaders:** Redpanda Connect/Bento use `golangci-lint` + `gofumpt` + a `Taskfile` with `task fmt`, `task lint`, `task test` ([Connect README](https://github.com/redpanda-data/connect)). Use `staticcheck` (golint is deprecated).
- **Notebooks are scratch, not production code** — the same rule the Python world converged on: real training/inference logic lives in versioned packages under `internal/` ([GitHub community discussion, Mar 2026](https://github.com/orgs/community/discussions/189034)).

---

## 4. Exemplary repositories to study (and what to steal from each)

| Repository | Role | What to study | License / vitality |
|---|---|---|---|
| **[warpstreamlabs/bento](https://github.com/warpstreamlabs/bento)** | Declarative stream processor (Benthos fork) | **#1 architecture reference for data pipelines.** Plugin/connector interfaces, stateless processor stages, at-least-once delivery without persisted state, config-driven design, clean `Taskfile`/lint setup | MIT; v1.21.1 Aug 2026, ~2.2k stars, commits through Oct 2026 |
| [redpanda-data/connect](https://github.com/redpanda-data/connect) | Same lineage, vendor-backed distribution | 300+ connectors (Kafka, S3, SQL, CDC for Postgres/MySQL/Mongo…), Bloblang mapping, huge integration-test matrix; compare its split-repo structure with Bento | Apache-2.0 connectors (Snowflake/Splunk enterprise-gated), MIT core; ~8.8k stars, active Oct 2026 |
| **[gomlx/gomlx](https://github.com/gomlx/gomlx)** | Modern accelerated ML framework | How to structure a serious ML library in Go: backend abstraction (`xla` vs pure-Go), context-based graph building, autodiff, ONNX interop, GoNB-integrated tutorials | Apache-2.0; v0.28.x 2026, ~1.6k stars, active |
| [gonum/gonum](https://github.com/gonum/gonum) | Numerical computing | Mature package boundaries (`mat`, `stat`, `blas`, `optimize`, `plot`), test discipline, API stability practices | BSD-3; the bedrock nearly everything else builds on |
| [gorgonia/gorgonia](https://github.com/gorgonia/gorgonia) | Legacy deep-learning graph engine | Expression-graph + autodiff design; worth reading for concepts, not adopting | Apache-2.0; ~5.9k stars but >1 year since meaningful activity |
| [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) + [microsoft/onnxruntime/go](https://github.com/microsoft/onnxruntime/go) | ONNX inference bindings | Tensor/session lifecycle, CGO shared-library handling, the realistic way to serve Python-trained models in Go | MIT / MIT; official bindings new in 2026 |
| [knights-analytics/hugot](https://github.com/knights-analytics/hugot) | Transformer pipelines in Go | Production NLP/vision pipeline packaging around ONNX with swappable backends | Active, check repo for current license |
| **[influxdata/telegraf](https://github.com/influxdata/telegraf)** | Data-collection agent | **Plugin-interface masterclass:** four tiny interfaces — Input `Gather()`, Processor `Apply()`, Aggregator `Add()/Push()`, Output `Write()` — that scaled to 300+ plugins and 1,200+ contributors; single static binary, TOML config | MIT; 17.7k stars, v1.39.1 Jun 2026 |
| [dolthub/dolt](https://github.com/dolthub/dolt) | Version-controlled database | Large-scale Go monorepo organization, content-addressed storage (Prolly Trees), CLI + server multi-binary layout | Apache-2.0; 24.7k stars, active Oct 2026 |
| [janpfeifer/gonb](https://github.com/janpfeifer/gonb) | Jupyter kernel for Go | How to keep an exploratory workflow without leaving the Go ecosystem; pair with `go-plotly` for charts | Active through Jul 2026 |
| [mioluigi/go-onnx-production](https://github.com/mioluigi/go-onnx-production) | End-to-end reference project | A complete worked example: PyTorch → ONNX → Go serving, with parity tests (10k MNIST images, probabilities within ~1e-6) and honest latency benchmarks | Small but directly copyable |

---

## 5. Pick-by-scenario decision guide

| Your situation | Do this |
|---|---|
| **ETL / streaming data pipeline** (the most common Go data use) | Build on Bento as a library/SDK or copy its plugin pattern; CLI with `spf13/cobra`; config in YAML/TOML; study Telegraf's interface design |
| **Serve existing ML models at low latency/high throughput** | Train in Python, export ONNX, serve with `yalue/onnxruntime_go` (stable today) or official `microsoft/onnxruntime/go` (watch for maturation); use `go-onnx-production` as the reference implementation |
| **NLP/transformer features inside a Go service** | Hugot (ONNX transformer pipelines) rather than rolling your own tokenization/batching |
| **You want to train models natively in Go** | GoMLX (XLA backend for GPU/TPU, pure-Go backend for portability) — accept pre-1.0 API churn and pin versions; Gorgonia only for legacy codebases |
| **Datasets need branching/audit/rollback** | Dolt for data version control; DVC if you stay file-based |
| **Exploratory analysis** | GoNB + Gonum/`go-plotly`; if the team already knows Python/Jupyter, a hybrid repo (Python `notebooks/` + Go `cmd/` services) is pragmatic and common |

### Honest caveat

If the project's core is *model research and experimentation* (rapid prototyping, GPU-heavy training, notebook-driven iteration), Python remains materially more productive — Go's dataframe and tensor ecosystems are smaller and its notebook story is younger ([golangprojectstructure.com](https://golangprojectstructure.com/can-go-be-used-for-machine-learning/), [Quora discussion](https://www.quora.com/Is-Go-the-best-programming-language-for-next-generation-A-I-and-machine-learning-applications)). The strongest repositories treat Go as the **productionization layer**: pipelines, feature services, inference servers, agents.

---

## 6. Recommended next steps

1. Scaffold with `gonew` (or the AI-pipeline template) using the §3.3 layout — start with just `cmd/serve` + `internal/`, add directories as code arrives.
2. Build a vertical slice first: one ONNX model served via a Go HTTP endpoint, parity-tested against Python outputs (mirror `go-onnx-production`).
3. Adopt the Bento/Telegraf plugin-interface pattern before writing a second connector or processor.
4. Wire CI on day one: `gofumpt`, `golangci-lint`, `staticcheck`, `go test -race`, coverage — the exact bar the reference repos set.
5. Decide artifact strategy early: DVC/LFS/object storage for models; Dolt if data versioning is a product requirement.

---

## Sources

- [Go Project Structure 2026 — levelupgo.dev](https://levelupgo.dev/blog/go-project-structure) · [No-nonsense Go layout — daveamit.com](https://daveamit.com/posts/2026-02-13-go-folder-structure/) · [golang-standards/project-layout](https://github.com/golang-standards/project-layout) · [gonew — go.dev blog](https://go.dev/blog/gonew) · [project-layout best practices — DeepWiki](https://deepwiki.com/golang-standards/project-layout/8-best-practices)
- [Go AI-driven pipeline template — link-foundation](https://github.com/link-foundation/go-ai-driven-development-pipeline-template) · [Structuring AI/ML projects — GitHub Community](https://github.com/orgs/community/discussions/189034)
- [Golang in the AI/ML Landscape: A Practical Guide — Medium](https://medium.com/@vladimirvivien/go-in-the-ai-ml-landscape-a-practical-guide-d36d44f360d2) · [awesome-golang-ai](https://github.com/promacanthus/awesome-golang-ai) · [gopherdata/resources](https://github.com/gopherdata/resources/blob/master/tooling/README.md)
- [GoMLX GitHub](https://github.com/gomlx/gomlx) · [GoMLX site](https://gomlx.github.io/) · [GoMLX: ML in Go without Python — Eli Bendersky](https://eli.thegreenplace.net/2024/gomlx-ml-in-go-without-python/) · [Gorgonia vs goml — libhunt](https://www.libhunt.com/compare-gorgonia-vs-goml)
- [Redpanda Connect GitHub](https://github.com/redpanda-data/connect) · [Bento GitHub](https://github.com/warpstreamlabs/bento) · [Announcing Bento — WarpStream](https://www.warpstream.com/blog/announcing-bento-the-open-source-fork-of-the-project-formerly-known-as-benthos) · [Redpanda Connect launch](https://www.redpanda.com/blog/redpanda-connect) · [TechCrunch acquisition coverage](https://techcrunch.com/2024/05/30/redpanda-acquires-benthos-to-expand-its-end-to-end-streaming-data-platform/)
- [onnxruntime_go — yalue](https://github.com/yalue/onnxruntime_go) · [ML Serving with Golang — Medium](https://bgweber.medium.com/options-for-ml-serving-with-golang-3b1afa3a6182) · [go-onnx-production](https://github.com/mioluigi/go-onnx-production) · [Hugot](https://github.com/knights-analytics/hugot)
- [Telegraf GitHub](https://github.com/influxdata/telegraf) · [Telegraf facts — dev.co](https://dev.co/observability/open-source/telegraf) · [Telegraf architecture — InfluxData](https://www.influxdata.com/blog/telegraf-go-collection-agent/)
- [DoltHub GitHub](https://github.com/dolthub) · [Dolt 2.0 — InfoQ](https://www.infoq.com/news/2026/07/dolt-version-control/) · [GoNB GitHub](https://github.com/janpfeifer/gonb)
- [Can Go be used for ML? — golangprojectstructure.com](https://golangprojectstructure.com/can-go-be-used-for-machine-learning/) · [Golang in AI: applications and libraries — Calmops](https://calmops.com/programming/golang/golang-in-ai-applications-and-libraries/)
