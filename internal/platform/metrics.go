package platform

import (
	"fmt"
	"io"
	"sort"
	"sync"
)

// Metrics is a tiny concurrency-safe registry of counters and gauges that
// renders the Prometheus text exposition format. It deliberately avoids a
// client-library dependency; swap in prometheus/client_golang when the
// service outgrows counters like these.
type Metrics struct {
	mu       sync.Mutex
	counters map[string]float64
	gauges   map[string]float64
}

// NewMetrics returns an empty registry.
func NewMetrics() *Metrics {
	return &Metrics{counters: map[string]float64{}, gauges: map[string]float64{}}
}

// Inc adds 1 to a counter. labelKey/labelValue may be empty.
func (m *Metrics) Inc(name, labelKey, labelValue string) { m.Add(name, labelKey, labelValue, 1) }

// Add adds delta to a counter.
func (m *Metrics) Add(name, labelKey, labelValue string, delta float64) {
	k := seriesKey(name, labelKey, labelValue)
	m.mu.Lock()
	m.counters[k] += delta
	m.mu.Unlock()
}

// Set records a gauge value.
func (m *Metrics) Set(name string, value float64) {
	m.mu.Lock()
	m.gauges[seriesKey(name, "", "")] = value
	m.mu.Unlock()
}

// WritePrometheus writes all series in Prometheus text format.
func (m *Metrics) WritePrometheus(w io.Writer) {
	m.mu.Lock()
	defer m.mu.Unlock()
	keys := make([]string, 0, len(m.counters)+len(m.gauges))
	for k := range m.counters {
		keys = append(keys, "counter\x00"+k)
	}
	for k := range m.gauges {
		keys = append(keys, "gauge\x00"+k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		kind, series := k[:7], k[8:]
		typ, val := "counter", m.counters[series]
		if kind == "gauge\x00" {
			typ, val = "gauge", m.gauges[series]
		}
		name, labels := splitSeries(series)
		fmt.Fprintf(w, "# TYPE %s %s\n%s%s %v\n", name, typ, name, labels, val)
	}
}

func seriesKey(name, labelKey, labelValue string) string {
	if labelKey == "" {
		return name
	}
	return fmt.Sprintf("%s{%s=%q}", name, labelKey, labelValue)
}

func splitSeries(series string) (name, labels string) {
	for i := 0; i < len(series); i++ {
		if series[i] == '{' {
			return series[:i], series[i:]
		}
	}
	return series, ""
}
