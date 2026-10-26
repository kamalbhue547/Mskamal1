# Serving real ONNX models from Go

The production pattern this repo is built around (research §1, §5):
**train in Python → export ONNX → serve in Go.** The demo model is a
softmax-regression classifier, but the serving path is the real thing:
`yalue/onnxruntime_go` (the mature community binding used with ONNX Runtime
1.31 here; Microsoft's official Go bindings are the ones to watch — research §2).

## 1. Get the artifacts

Either train in Go (pure-Go artifact only):

```bash
make run-pipeline          # writes models/demo/model.json
```

…or train in Python and export both artifacts (recommended — this is the
path that exercises ONNX):

```bash
pip install numpy onnx onnxruntime
make train-onnx            # scripts/train_export_onnx.py
# -> models/demo/model.json  (shared schema; served by the pure-Go backend)
# -> models/demo/model.onnx  (scaler baked into the graph; ONNX backend)
```

The two files are a matched pair and must be deployed together: `model.json`
is the source of truth for class-label order, `model.onnx` for the compute.

## 2. Get the ONNX Runtime shared library

```bash
scripts/download_onnxruntime.sh            # -> third_party/onnxruntime/
export ONNX_LIB_PATH="$PWD/third_party/onnxruntime/lib/libonnxruntime.so"
```

No download needed if Python is around — the wheel already ships it:

```bash
export ONNX_LIB_PATH="$(python3 -c 'import onnxruntime,os,glob; print(glob.glob(os.path.join(os.path.dirname(onnxruntime.__file__),"capi","libonnxruntime.so*"))[0])')"
```

## 3. Verify parity before serving (the important step)

The pure-Go predictor and the ONNX predictor must agree on the same weights —
this is the test borrowed from `go-onnx-production` (research §4):

```bash
make onnx-test ONNX_LIB="$ONNX_LIB_PATH"
# TestONNXParity: cluster centres + boundary points, probabilities within 1e-5
```

## 4. Build and serve with the ONNX backend

```bash
make build-onnx
./bin/serve -model models/demo/model.json \
            -onnx models/demo/model.onnx \
            -onnx-lib "$ONNX_LIB_PATH" -addr :8080

curl localhost:8080/v1/model
# {"name":"demo-softmax","version":"1.0.0","backend":"onnx",...}

curl -X POST localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[2,-1,2,-1]}'
# {"label":"class_c","class_index":2,"probabilities":[...],"backend":"onnx",...}
```

Omitting `-onnx` serves the same `model.json` through the pure-Go backend
(`"backend":"go"`) — same request, same response shape, no CGO.

## Graph contract

Fixed between `scripts/train_export_onnx.py` and `internal/inference/onnx.go`:

| Tensor | Name | Type | Shape |
|---|---|---|---|
| input | `features` | float32 | `[1, n_features]` — **raw**, unscaled |
| output | `probabilities` | float32 | `[1, n_classes]` — softmax |
| output | `label` | int64 | `[1]` — argmax |

Standardisation fitted in Python is baked into the exported Gemm
(`W' = W/scale`, `b' = b − (mean/scale)·W`), so callers never scale by hand
in either backend.

## Serving a different model

1. Export it with the same tensor contract (or extend `onnx.go`'s
   input/output names and shapes — they are the only backend-specific lines).
2. Write/keep a `model.json` with matching `feature_names` and `classes`
   order (the exporter already does this for the demo model).
3. Re-run the parity test against a reference implementation before deploy.

For HuggingFace transformer pipelines, don't hand-roll tokenisation in Go —
use Hugot (research §5) behind the same `inference.Predictor` interface.
