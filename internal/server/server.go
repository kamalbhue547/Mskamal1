// Package server is the HTTP serving layer for one loaded Predictor.
// Routing uses only the standard library (Go 1.22+ method patterns), keeping
// the pure-Go binary dependency-free and trivially embeddable in tests.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/kamalbhue547/mskamal1/internal/inference"
	"github.com/kamalbhue547/mskamal1/internal/platform"
)

// Server wires a Predictor to HTTP handlers with metrics and logging.
type Server struct {
	predictor inference.Predictor
	metrics   *platform.Metrics
	logger    *slog.Logger
}

// New builds a Server. Nil metrics/logger get usable defaults.
func New(p inference.Predictor, m *platform.Metrics, logger *slog.Logger) *Server {
	if m == nil {
		m = platform.NewMetrics()
	}
	if logger == nil {
		logger = platform.NewLogger("info")
	}
	return &Server{predictor: p, metrics: m, logger: logger}
}

// Handler returns the full route set (documented in api/openapi.yaml).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /readyz", s.handleHealth)
	mux.HandleFunc("GET /v1/model", s.handleModel)
	mux.HandleFunc("POST /v1/predict", s.handlePredict)
	mux.Handle("GET /metrics", http.HandlerFunc(s.handleMetrics))
	return s.middleware(mux)
}

// Serve runs an HTTP server with graceful shutdown on ctx cancellation.
func (s *Server) Serve(ctx context.Context, addr string, readTimeout, writeTimeout, shutdownTimeout time.Duration) error {
	srv := &http.Server{
		Addr:         addr,
		Handler:      s.Handler(),
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
	}
	errCh := make(chan error, 1)
	go func() {
		s.logger.Info("serving", "addr", addr, "model", s.predictor.Info().Name, "backend", s.predictor.Info().Backend)
		errCh <- srv.ListenAndServe()
	}()
	select {
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown: %w", err)
		}
		return nil
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleModel(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.predictor.Info())
}

func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.metrics.WritePrometheus(w)
}

// PredictRequest is the POST /v1/predict body. Features are raw (unscaled);
// scaling is the model's job, identically in every backend.
type PredictRequest struct {
	Features []float64 `json:"features"`
}

// PredictResponse is the POST /v1/predict success body.
type PredictResponse struct {
	Label         string    `json:"label"`
	ClassIndex    int       `json:"class_index"`
	Probabilities []float64 `json:"probabilities"`
	Model         string    `json:"model"`
	Backend       string    `json:"backend"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	if err != nil {
		s.fail(w, http.StatusRequestEntityTooLarge, "request body too large or unreadable")
		return
	}
	var req PredictRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.fail(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	info := s.predictor.Info()
	if len(req.Features) != len(info.FeatureNames) {
		s.fail(w, http.StatusBadRequest,
			fmt.Sprintf("expected %d features %v, got %d", len(info.FeatureNames), info.FeatureNames, len(req.Features)))
		return
	}
	pred, err := s.predictor.Predict(r.Context(), req.Features)
	if err != nil {
		s.logger.Error("predict failed", "error", err)
		s.metrics.Inc("predict_errors_total", "", "")
		s.fail(w, http.StatusInternalServerError, "prediction failed")
		return
	}
	s.metrics.Inc("predictions_total", "label", pred.Label)
	writeJSON(w, http.StatusOK, PredictResponse{
		Label:         pred.Label,
		ClassIndex:    pred.ClassIndex,
		Probabilities: pred.Probabilities,
		Model:         info.Name,
		Backend:       info.Backend,
	})
}

func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer func() {
			if rec := recover(); rec != nil {
				s.logger.Error("panic in handler", "panic", rec, "path", r.URL.Path)
				s.fail(w, http.StatusInternalServerError, "internal error")
			}
		}()
		next.ServeHTTP(w, r)
		s.metrics.Inc("http_requests_total", "path", r.URL.Path)
		s.logger.Debug("request", "method", r.Method, "path", r.URL.Path, "duration_ms", time.Since(start).Milliseconds())
	})
}

func (s *Server) fail(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
