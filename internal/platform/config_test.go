package platform

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadServeConfigDefaultsAndEnv(t *testing.T) {
	t.Setenv("SERVE_ADDR", ":9090")
	cfg, err := LoadServeConfig("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Addr != ":9090" {
		t.Fatalf("addr = %q, want :9090 (env override)", cfg.Addr)
	}
	if cfg.ModelPath != "models/demo/model.json" {
		t.Fatalf("unexpected default model path %q", cfg.ModelPath)
	}
}

func TestLoadServeConfigFileAndUnknownField(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "serve.json")
	if err := os.WriteFile(good, []byte(`{"addr":":8081","model_path":"m.json"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadServeConfig(good)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Addr != ":8081" || cfg.ModelPath != "m.json" {
		t.Fatalf("cfg = %+v", cfg)
	}

	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"nope":1}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadServeConfig(bad); err == nil {
		t.Fatal("expected error for unknown field")
	}
}

func TestLoadPipelineConfigValidation(t *testing.T) {
	if _, err := LoadPipelineConfig(""); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "p.json")
	if err := os.WriteFile(p, []byte(`{"epochs":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPipelineConfig(p); err == nil {
		t.Fatal("expected epochs validation error")
	}
}
