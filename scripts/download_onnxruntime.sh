#!/usr/bin/env bash
# Download the ONNX Runtime shared library for the CGO serving build.
#
#   scripts/download_onnxruntime.sh [VERSION]     # default 1.31.0
#   export ONNX_LIB_PATH="$PWD/third_party/onnxruntime/lib/libonnxruntime.so"
#   make build-onnx && make onnx-test ONNX_LIB="$ONNX_LIB_PATH"
#
# Alternative with zero download: if Python is installed, the onnxruntime
# wheel already ships the library —
#   python3 -c 'import onnxruntime,os; print(os.path.join(os.path.dirname(onnxruntime.__file__),"capi"))'
set -euo pipefail

VERSION="${1:-1.31.0}"
DEST="${2:-third_party/onnxruntime}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  PKG="onnxruntime-linux-x64-${VERSION}" ;;
  Linux-aarch64) PKG="onnxruntime-linux-aarch64-${VERSION}" ;;
  Darwin-arm64)  PKG="onnxruntime-osx-arm64-${VERSION}" ;;
  Darwin-x86_64) PKG="onnxruntime-osx-x86_64-${VERSION}" ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m); download manually from https://github.com/microsoft/onnxruntime/releases" >&2; exit 1 ;;
esac

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Downloading ${PKG}.tgz (ONNX Runtime ${VERSION})..."
curl -fsSL -o "$TMP/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.tgz"
tar -xzf "$TMP/ort.tgz" -C "$TMP"
cp -R "$TMP/${PKG}/lib" "$TMP/${PKG}/include" "$DEST/"
echo "Installed to $DEST"
case "$(uname -s)" in
  Darwin) LIB="$DEST/lib/libonnxruntime.dylib" ;;
  *)      LIB="$DEST/lib/libonnxruntime.so" ;;
esac
echo "export ONNX_LIB_PATH=\"$PWD/$LIB\""
