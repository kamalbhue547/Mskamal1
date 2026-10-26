package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kamalbhue547/mskamal1/internal/inference"
	"github.com/kamalbhue547/mskamal1/internal/ingest"
	"github.com/kamalbhue547/mskamal1/internal/model"
)

func testServer(t *testing.T) http.Handler {
	t.Helper()
	ds, err := ingest.SyntheticSource{Rows: 300, Seed: 42}.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	train, _ := model.SplitTrainTest(ds, 0.2, 42)
	a, err := model.Train(train, model.TrainOptions{Epochs: 300, LearningRate: 0.1, L2: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	p, err := inference.NewArtifactPredictor(a)
	if err != nil {
		t.Fatal(err)
	}
	return New(p, nil, nil).Handler()
}

func TestHealthAndModelEndpoints(t *testing.T) {
	h := testServer(t)
	for _, path := range []string{"/healthz", "/readyz", "/v1/model", "/metrics"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s: status %d, body %s", path, rec.Code, rec.Body.String())
		}
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/model", nil))
	var info inference.ModelInfo
	if err := json.Unmarshal(rec.Body.Bytes(), &info); err != nil {
		t.Fatal(err)
	}
	if info.Backend != "go" || len(info.Classes) != 3 {
		t.Fatalf("info = %+v", info)
	}
}

func TestPredictEndpoint(t *testing.T) {
	h := testServer(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", strings.NewReader(`{"features":[-2,-2,-2,-2]}`))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, body %s", rec.Code, rec.Body.String())
	}
	var resp PredictResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Label != "class_a" || len(resp.Probabilities) != 3 {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestPredictValidatesInput(t *testing.T) {
	h := testServer(t)
	for _, body := range []string{`{"features":[1,2]}`, `not json`, `{}`} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/predict", strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("body %q: status %d, want 400", body, rec.Code)
		}
	}
}
