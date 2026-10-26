// Package client is the one deliberately public package in this repository
// (see docs/architecture.md on why pkg/ stays nearly empty): a small HTTP
// client for the serving API so other Go services can consume predictions
// without copying DTOs.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client talks to a cmd/serve instance.
type Client struct {
	BaseURL string
	HTTP    *http.Client // nil means a 10s-timeout default
}

// New returns a Client for baseURL like "http://localhost:8080".
func New(baseURL string) *Client { return &Client{BaseURL: baseURL} }

// ModelInfo mirrors GET /v1/model.
type ModelInfo struct {
	Name         string   `json:"name"`
	Version      string   `json:"version"`
	Backend      string   `json:"backend"`
	FeatureNames []string `json:"feature_names"`
	Classes      []string `json:"classes"`
}

// Prediction mirrors POST /v1/predict.
type Prediction struct {
	Label         string    `json:"label"`
	ClassIndex    int       `json:"class_index"`
	Probabilities []float64 `json:"probabilities"`
	Model         string    `json:"model"`
	Backend       string    `json:"backend"`
}

func (c *Client) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 10 * time.Second}
}

// Health reports whether the server answers GET /healthz.
func (c *Client) Health(ctx context.Context) error {
	resp, err := c.do(ctx, http.MethodGet, "/healthz", nil)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health: status %d", resp.StatusCode)
	}
	return nil
}

// Model returns the loaded model metadata.
func (c *Client) Model(ctx context.Context) (*ModelInfo, error) {
	resp, err := c.do(ctx, http.MethodGet, "/v1/model", nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model: status %d", resp.StatusCode)
	}
	var info ModelInfo
	return &info, json.NewDecoder(resp.Body).Decode(&info)
}

// Predict sends raw features and returns the prediction.
// features must have the length reported by Model().FeatureNames.
func (c *Client) Predict(ctx context.Context, features []float64) (*Prediction, error) {
	body, err := json.Marshal(map[string][]float64{"features": features})
	if err != nil {
		return nil, err
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/predict", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&e)
		return nil, fmt.Errorf("predict: status %d: %s", resp.StatusCode, e.Error)
	}
	var pred Prediction
	return &pred, json.NewDecoder(resp.Body).Decode(&pred)
}

func (c *Client) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, rdr)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return c.httpClient().Do(req)
}
