# Results

Raw records are committed. Aggregates are not.

- `<zoo>/report.md` and `<zoo>/aggregate.json` are **generated**, always from
  the raw JSONL under `$DEADBOLT_ROOT/zoo/<zoo>/`. Never hand-edit them; run
  `deadbolt report --zoo <zoo>` instead.
- `<zoo>/figures/` is gitignored. Plots regenerate from the same records.

The point of the split is that every number in a report can be traced back to a
raw scan row carrying the git commit, seed, and config hash that produced it.
An aggregate that cannot be rebuilt from raw records is an assertion, not a
result.
