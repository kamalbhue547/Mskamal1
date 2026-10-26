package pipeline

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunnerRetriesThenSucceeds(t *testing.T) {
	calls := 0
	flaky := StageFunc{StageName: "flaky", Fn: func(context.Context) error {
		calls++
		if calls < 3 {
			return errors.New("boom")
		}
		return nil
	}}
	r := NewRunner(Options{MaxAttempts: 3, Backoff: time.Millisecond})
	if err := r.Run(context.Background(), flaky); err != nil {
		t.Fatal(err)
	}
	if len(r.Results) != 1 || r.Results[0].Attempts != 3 {
		t.Fatalf("results = %+v", r.Results)
	}
}

func TestRunnerStopsAtFirstFailure(t *testing.T) {
	ran := false
	r := NewRunner(Options{})
	err := r.Run(context.Background(),
		StageFunc{StageName: "bad", Fn: func(context.Context) error { return errors.New("nope") }},
		StageFunc{StageName: "never", Fn: func(context.Context) error { ran = true; return nil }},
	)
	if err == nil {
		t.Fatal("expected error")
	}
	if ran {
		t.Fatal("stage after failure must not run")
	}
}
