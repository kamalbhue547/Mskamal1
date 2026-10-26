# Multi-stage build for the pure-Go serving binary (no CGO, no ONNX Runtime).
# For the ONNX variant, build locally with `make build-onnx` — it needs the
# ONNX Runtime shared library at both build and run time; see
# docs/onnx-serving.md. Keeping the default image CGO-free keeps it tiny and
# is exactly why the default backend exists.
FROM golang:1.26-bookworm AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/ ./cmd/...

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /out/serve /serve
COPY configs/serve.json /configs/serve.json
COPY models/demo/model.json /models/demo/model.json
EXPOSE 8080
ENTRYPOINT ["/serve", "-config", "/configs/serve.json"]
