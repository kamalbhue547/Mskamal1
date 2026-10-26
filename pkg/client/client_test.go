package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientAgainstStubServer(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /v1/model", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(ModelInfo{Name: "demo-softmax", Backend: "go", FeatureNames: []string{"a"}, Classes: []string{"x", "y"}})
	})
	mux.HandleFunc("POST /v1/predict", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(Prediction{Label: "x", ClassIndex: 0, Probabilities: []float64{0.9, 0.1}, Model: "demo-softmax", Backend: "go"})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := New(srv.URL)
	ctx := context.Background()
	if err := c.Health(ctx); err != nil {
		t.Fatal(err)
	}
	info, err := c.Model(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if info.Name != "demo-softmax" {
		t.Fatalf("info = %+v", info)
	}
	pred, err := c.Predict(ctx, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if pred.Label != "x" || pred.Probabilities[0] != 0.9 {
		t.Fatalf("pred = %+v", pred)
	}
}
