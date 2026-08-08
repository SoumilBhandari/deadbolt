# Results

Every number here must be rebuildable from inputs that are also here.

Each `<zoo>/` directory holds four files, all written by
`deadbolt report --zoo <zoo>`:

| File | What it is |
|---|---|
| `scans.jsonl` | **Raw input.** One row per detector run, latest-wins per `(checkpoint, defense)`, each carrying the git commit, seed, and config hash that produced it. |
| `manifest.jsonl` | **Raw input.** One row per trained model — the poisoning ground truth, including runs filtered out with their reason. |
| `report.md` | Generated. The human-readable tables. |
| `aggregate.json` | Generated. The same numbers, machine-readable. |

`<zoo>/figures/` is gitignored; plots regenerate from the same records.

Never hand-edit any of them — rerun `deadbolt report --zoo <zoo>` instead.

## Why the raw records are committed too

The two `.jsonl` files are the point. The zoo they were produced from lives
under `$DEADBOLT_ROOT`, which is machine-local and gitignored — it runs to
gigabytes of checkpoints. If only `report.md` and `aggregate.json` were
committed, the tables would be conclusions whose inputs nobody else has, not
even the author on a different machine. Anyone could read the numbers; nobody
could check them.

With the raw rows committed, every figure in a report traces back to a scan
row that names the commit, seed, and config that produced it, and the whole
aggregate can be rebuilt by anyone who clones the repo.

An aggregate that cannot be rebuilt from raw records is an assertion, not a
result. That standard applies to our own numbers first.

## Rebuilding

```bash
deadbolt report --zoo tierA
```

Regenerates `report.md` and `aggregate.json` from `scans.jsonl` and
`manifest.jsonl`. A clean `git diff` afterwards means the committed tables
match the committed records; a dirty one means they had drifted.
