// Package pipeline runs ordered stages with retry semantics — a deliberately
// minimal DAG (a list) until a real branching graph is needed. Each stage is
// idempotent by convention: re-running the pipeline must be safe, which is
// what makes retries and manual re-runs boring instead of dangerous.
package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// Stage is one unit of pipeline work.
type Stage interface {
	Name() string
	Run(ctx context.Context) error
}

// StageFunc adapts a plain function to a Stage.
type StageFunc struct {
	StageName string
	Fn        func(ctx context.Context) error
}

func (s StageFunc) Name() string { return s.StageName }

// Run implements Stage.
func (s StageFunc) Run(ctx context.Context) error { return s.Fn(ctx) }

// Result records the outcome of one stage execution.
type Result struct {
	Stage    string
	Attempts int
	Duration time.Duration
	Err      error
}

// Options controls runner behaviour.
type Options struct {
	MaxAttempts int           // total attempts per stage (default 1 = no retry)
	Backoff     time.Duration // base delay between attempts, doubled each retry
	Logger      *slog.Logger  // may be nil
}

// Runner executes stages sequentially, stopping at the first failure.
type Runner struct {
	opts    Options
	Results []Result
}

// NewRunner builds a Runner; zero MaxAttempts means a single attempt.
func NewRunner(opts Options) *Runner {
	if opts.MaxAttempts < 1 {
		opts.MaxAttempts = 1
	}
	return &Runner{opts: opts}
}

// Run executes stages in order and returns the first stage error (wrapped).
// Per-stage results are accumulated on the Runner for reporting.
func (r *Runner) Run(ctx context.Context, stages ...Stage) error {
	for _, stage := range stages {
		res := Result{Stage: stage.Name()}
		start := time.Now()
		for attempt := 1; attempt <= r.opts.MaxAttempts; attempt++ {
			res.Attempts = attempt
			res.Err = stage.Run(ctx)
			if res.Err == nil {
				break
			}
			if r.opts.Logger != nil {
				r.opts.Logger.Warn("stage failed, will retry",
					"stage", stage.Name(), "attempt", attempt, "error", res.Err)
			}
			if attempt < r.opts.MaxAttempts && r.opts.Backoff > 0 {
				select {
				case <-ctx.Done():
					res.Err = ctx.Err()
				case <-time.After(r.opts.Backoff << (attempt - 1)):
				}
				if res.Err == ctx.Err() && ctx.Err() != nil {
					break
				}
			}
		}
		res.Duration = time.Since(start)
		r.Results = append(r.Results, res)
		if r.opts.Logger != nil {
			r.opts.Logger.Info("stage finished",
				"stage", res.Stage, "attempts", res.Attempts,
				"duration_ms", res.Duration.Milliseconds(), "error", res.Err)
		}
		if res.Err != nil {
			return fmt.Errorf("pipeline stage %q failed after %d attempt(s): %w", res.Stage, res.Attempts, res.Err)
		}
	}
	return nil
}
