package platform

import (
	"strings"
	"testing"
)

func TestMetricsWritePrometheus(t *testing.T) {
	m := NewMetrics()
	m.Inc("predictions_total", "label", "class_a")
	m.Inc("predictions_total", "label", "class_a")
	m.Add("predictions_total", "label", "class_b", 3)
	m.Set("model_loaded", 1)

	var sb strings.Builder
	m.WritePrometheus(&sb)
	out := sb.String()
	for _, want := range []string{
		"# TYPE predictions_total counter",
		`predictions_total{label="class_a"} 2`,
		`predictions_total{label="class_b"} 3`,
		"# TYPE model_loaded gauge",
		"model_loaded 1",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics output missing %q:\n%s", want, out)
		}
	}
}
