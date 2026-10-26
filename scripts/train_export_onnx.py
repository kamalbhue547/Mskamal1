#!/usr/bin/env python3
"""Train the demo model in Python and export it for serving from Go.

This is the Python side of the repository's central workflow
(docs/go-data-ml-research.md §1): train in Python -> export ONNX -> serve in
Go. It intentionally uses only numpy + onnx (no scikit-learn / torch needed
for a softmax-regression demo), and it writes two artifacts that must always
travel together:

  model.json  -- the exact schema internal/model loads in Go
                 (scaler + weights), served by the pure-Go predictor.
  model.onnx  -- the same model as an ONNX graph for the CGO/ONNX Runtime
                 predictor (-tags onnx). The fitted standardisation is baked
                 into the first Gemm, so the graph consumes *raw* features,
                 exactly like model.json does. internal/inference's
                 TestONNXParity proves the two agree to float32 tolerance.

Usage:
    python3 scripts/train_export_onnx.py --data data/raw/sample.csv \
        --out-dir models/demo
    make onnx-test ONNX_LIB=/path/to/libonnxruntime.so
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as exc:  # pragma: no cover - operator guidance
    sys.exit(f"missing dependency ({exc.name}); run: pip install numpy onnx onnxruntime")


def load_csv(path: Path) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"{path}: need a header and at least one record")
    feature_names, label_name = rows[0][:-1], rows[0][-1]
    classes: list[str] = []
    x, labels = [], []
    for row in rows[1:]:
        x.append([float(v) for v in row[:-1]])
        label = row[-1]
        if label not in classes:  # first-seen order, same as Go's Classes()
            classes.append(label)
        labels.append(label)
    y = np.array([classes.index(v) for v in labels], dtype=np.int64)
    return feature_names, np.array(x, dtype=np.float64), y, classes


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train(x: np.ndarray, y: np.ndarray, n_classes: int, epochs: int, lr: float, l2: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)  # population std (ddof=0), matching Go
    scale = np.where(scale == 0, 1.0, scale)
    xs = (x - mean) / scale

    n, n_feat = xs.shape
    w = np.zeros((n_feat, n_classes))
    b = np.zeros(n_classes)
    onehot = np.eye(n_classes)[y]
    for _ in range(epochs):
        probs = softmax(xs @ w + b)
        grad = (probs - onehot) / n
        w -= lr * (xs.T @ grad + l2 * w)
        b -= lr * grad.sum(axis=0)

    acc = float((np.argmax(xs @ w + b, axis=1) == y).mean())
    return mean, scale, w, b, acc


def export_onnx(path: Path, mean, scale, w, b, n_feat: int, n_classes: int) -> None:
    # Bake standardisation into the Gemm:
    #   logits = ((x-mean)/scale) @ W + b  ==  x @ (W/scale) + (b - (mean/scale) @ W)
    w_baked = (w / scale[:, None]).astype(np.float32)
    b_baked = (b - (mean / scale) @ w).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gemm", ["features", "W", "b"], ["logits"], alpha=1.0, beta=1.0),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
            helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0),
        ],
        name="demo-softmax",
        inputs=[helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_feat])],
        outputs=[
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, n_classes]),
            helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
        ],
        initializer=[
            numpy_helper.from_array(w_baked, name="W"),
            numpy_helper.from_array(b_baked, name="b"),
        ],
    )
    model = helper.make_model(graph, producer_name="mskamal1", opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/sample.csv", help="training CSV (features...,label)")
    ap.add_argument("--out-dir", default="models/demo", help="where model.json/model.onnx are written")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=0.001)
    args = ap.parse_args()

    feature_names, x, y, classes = load_csv(Path(args.data))
    mean, scale, w, b, acc = train(x, y, len(classes), args.epochs, args.lr, args.l2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "name": "demo-softmax",
        "version": "1.0.0",
        "task": "multiclass-classification",
        "feature_names": feature_names,
        "classes": classes,
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "weights": w.tolist(),  # weights[feature][class]
        "bias": b.tolist(),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {"train_accuracy": acc},
    }
    json_path = out_dir / "model.json"
    json_path.write_text(json.dumps(artifact, indent=2) + "\n")

    onnx_path = out_dir / "model.onnx"
    export_onnx(onnx_path, mean, scale, w, b, x.shape[1], len(classes))

    # Self-check with the real runtime when available: numpy vs ONNX must agree.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        centres = np.array([[-2, -2, -2, -2], [0, 2, 0, 2], [2, -1, 2, -1], [0, 0, 0, 0]], dtype=np.float32)
        probs_onnx, labels_onnx = sess.run(None, {"features": centres})
        probs_ref = softmax(((centres - mean) / scale) @ w + b)
        max_diff = float(np.abs(probs_onnx - probs_ref).max())
        assert max_diff < 1e-5, f"ONNX/numpy parity failed: max diff {max_diff}"
        print(f"self-check OK: onnx vs numpy max probability diff = {max_diff:.2e}, labels = {labels_onnx.tolist()}")
    except ImportError:
        print("onnxruntime not installed; skipped runtime self-check (pip install onnxruntime)")

    print(f"trained on {len(x)} rows, accuracy = {acc:.4f}, classes = {classes}")
    print(f"wrote {json_path} and {onnx_path}")


if __name__ == "__main__":
    main()
