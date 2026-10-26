# models/

Trained artifacts, one directory per model (`demo/` is the synthetic demo).

| File | Committed? | Why |
|---|---|---|
| `demo/model.json` | Yes — tiny JSON demo weights | Makes `make run-serve` work with zero setup; real model weights belong in object storage / DVC / Git LFS, not Git |
| `demo/model.onnx` | No (git-ignored) | Binary artifact; regenerate with `make train-onnx` or `scripts/download_onnxruntime.sh` + `docs/onnx-serving.md` |
| `demo/report.json` | No (git-ignored) | Per-run training report from `cmd/pipeline` |

`model.json` and `model.onnx` are a matched pair produced by the same run —
deploy them together. The schema is defined in `internal/model` and shared
with `scripts/train_export_onnx.py`.
