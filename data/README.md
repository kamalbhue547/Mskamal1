# data/

Datasets are **never committed** (see `.gitignore`) — only this README, the
`.gitkeep` files, and one deliberate exception: `raw/sample.csv`, the tiny
deterministic synthetic dataset that makes the demo reproducible.

| Directory | Contents | Produced by |
|---|---|---|
| `raw/` | Immutable originals as received | `cmd/ingest`, your collectors |
| `interim/` | Cleaned intermediate datasets | `cmd/ingest -input … -output data/interim/…` |
| `processed/` | Modeling-ready train/test splits | `cmd/pipeline` |

CSV format everywhere: a header of feature names followed by one label
column, one record per line. Regenerate the sample with `make run-ingest`.
For real data versioning (branch/audit/rollback), use Dolt or DVC — see
`docs/architecture.md`.
