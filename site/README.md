# ManipulaX results site

A single static page summarising the run: the demonstration clip, the per-seed
sub-task grid, the Intel device sweep, and the quantisation numbers.

Nothing on the page is typed by hand. `build_site.py` reads
`artifacts/eval/summary.json`, `artifacts/benchmark.json` and
`artifacts/accuracy.json` and emits `dist/index.html`, so the site cannot drift
away from the runs it describes — the same rule as
`scripts/make_submission_report.py`.

```bash
python site/build_site.py     # -> site/dist/index.html
```

`dist/` is committed so Vercel can deploy it directly (root directory `site/dist`,
no build step). Regenerate and commit it whenever the artifacts change.
