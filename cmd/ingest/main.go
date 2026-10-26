// Command ingest is the data-entry binary: Source → Processors → Sink.
//
//	# generate the deterministic synthetic demo dataset
//	go run ./cmd/ingest -output data/raw/sample.csv
//	# or clean an existing CSV into data/interim/
//	go run ./cmd/ingest -input data/raw/my.csv -output data/interim/my.csv
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/kamalbhue547/mskamal1/internal/ingest"
	"github.com/kamalbhue547/mskamal1/internal/platform"
)

func main() {
	input := flag.String("input", "", "input CSV (empty = generate synthetic demo data)")
	output := flag.String("output", "data/raw/sample.csv", "output .csv or .jsonl path")
	rows := flag.Int("rows", 300, "synthetic rows to generate (only with no -input)")
	seed := flag.Int64("seed", 42, "synthetic/shuffle seed")
	flag.Parse()

	logger := platform.NewLogger(os.Getenv("LOG_LEVEL"))
	ctx := context.Background()

	var src ingest.Source = ingest.SyntheticSource{Rows: *rows, Seed: *seed}
	if *input != "" {
		src = ingest.CSVSource{Path: *input}
	}

	ds, err := src.Read(ctx)
	if err != nil {
		log.Fatalf("ingest: read %s: %v", src.Name(), err)
	}
	ds, err = ingest.ApplyProcessors(ctx, ds, ingest.DropIncomplete{}, ingest.Shuffle(*seed))
	if err != nil {
		log.Fatalf("ingest: %v", err)
	}

	if dir := filepath.Dir(*output); dir != "." {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			log.Fatalf("ingest: %v", err)
		}
	}
	var sink ingest.Sink = ingest.CSVSink{Path: *output}
	if strings.HasSuffix(*output, ".jsonl") {
		sink = ingest.JSONLSink{Path: *output}
	}
	if err := sink.Write(ctx, ds); err != nil {
		log.Fatalf("ingest: write: %v", err)
	}
	logger.Info("ingest complete",
		"source", src.Name(), "sink", sink.Name(), "output", *output,
		"records", len(ds.Records), "classes", fmt.Sprint(ds.Classes()))
}
