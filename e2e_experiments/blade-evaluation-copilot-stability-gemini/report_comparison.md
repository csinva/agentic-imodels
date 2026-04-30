# Blade Stability Eval — Copilot CLI + Gemini 2.5 Pro

This folder mirrors `blade-evaluation-copilot-stability/` but drives the
Copilot CLI with Gemini 2.5 Pro instead of Claude Sonnet 4.5.

## Purpose

Control experiment: does the gain in the main sweep
(`blade-evaluation-copilot-gemini/`) come from naming the
`agentic_imodels` library specifically, or would *any* interpretable-ML
library reference produce a similar lift? Two near-identical prompts:

- **standard**: prompt mentions the `imodels` library
- **interpretml**: prompt mentions the `interpret` (InterpretML) library

Both prompts otherwise read the same. There is no `SKILL.md` or
custom package shipped to the agent — these are pure baselines.

## Setup

- Agent: `~/.local/bin/copilot-gemini -p ... --model gemini-2.5-pro
  --allow-all-tools --allow-all-paths --no-color`,
  hard-capped at `timeout 1200s` per dataset.
- Wrapper: invokes Copilot CLI's JS bundle via Node 24 because the precompiled
  Linux binary is a Node SEA whose hardcoded `l_n` blocklist hides Gemini
  models. The JS bundle's `app.js` was patched to remove `gemini-2.5-pro`
  from that set.
- Note: Gemini 3.1 (Preview) is not yet exposed in the Copilot CLI for this
  account; only `gemini-2.5-pro` is in the server-returned model list.
- Judge: Azure OpenAI `gpt-4o` via keyless Entra ID auth.
- Datasets: 13 Blade tasks (sourced via the sibling pipelines).
- Runs: 3 agent runs per mode × 3 judge runs = **9 evaluations per dataset
  per mode**. Total: 78 conclusions, 234 judge evaluations.
- Reliability: 78/78 cells produced conclusions; no timeouts in either mode.

## Results (mean ± SE across 13 datasets, 1–10 scale, n=9 evals/cell)

| Dimension     | standard         | interpretml      | Δ (int − std)    |
| ------------- | ---------------- | ---------------- | ---------------- |
| Correctness   | 4.70 ± 0.19      | 4.84 ± 0.12      | +0.14            |
| Completeness  | 3.71 ± 0.19      | 3.93 ± 0.13      | +0.22            |
| Clarity       | 4.44 ± 0.18      | 4.57 ± 0.14      | +0.14            |
| **Overall**   | **4.28**         | **4.45**         | **+0.17 (+3.9%)** |

**7 / 13 datasets favor interpretml**, 6 / 13 favor standard (imodels) —
toss-up. SE ranges overlap on every dimension.

## Per-Dataset Means (3 dims averaged, n=9)

| Dataset         | standard | interpretml |
|-----------------|----------|-------------|
| affairs         | 3.3      | 4.1         |
| amtl            | 4.3      | 3.8         |
| boxes           | 4.0      | 4.9         |
| caschools       | 4.5      | 4.2         |
| crofoot         | 3.4      | 4.7         |
| fertility       | 5.3      | 3.9         |
| fish            | 5.4      | 4.7         |
| hurricane       | 4.6      | 4.3         |
| mortgage        | 4.3      | 5.3         |
| panda_nuts      | 4.4      | 4.6         |
| reading         | 3.4      | 4.0         |
| soccer          | 4.3      | 4.2         |
| teachingratings | 4.8      | 4.9         |

## Comparison to agentic-imodels (`blade-evaluation-copilot-gemini/`)

| Setting (Gemini 2.5 Pro)                       | Overall mean |
| ---------------------------------------------- | ------------ |
| standard (imodels mentioned in prompt)         | 4.28         |
| interpretml (interpret mentioned in prompt)    | 4.45         |
| **custom_v2 (agentic_imodels + SKILL.md)**     | **7.62**     |

The two baseline prompts differ by 0.17 — within noise. The custom_v2 prompt
(which ships the evolved `agentic_imodels` package and a curated `SKILL.md`
analysis workflow) lifts the score by **+3.34** over either baseline. The
main-sweep gain is therefore *not* explained by "any interpretable-ML
library mentioned in the prompt"; it requires the `agentic_imodels` package
and the workflow guidance.

## Files

- `run_all.sh` — runs `copilot-gemini -p` per dataset (timeout 20 min).
- `prepare_run.py` — builds per-dataset run dirs (modes: standard /
  interpretml / custom_v2). Only standard + interpretml are used in this
  stability experiment.
- `evaluate.py` — Azure OpenAI LLM-as-a-judge.
- `outputs_standard_run{1,2,3}/`, `outputs_interpretml_run{1,2,3}/` —
  per-dataset working dirs with `analysis.py`, `conclusion.txt`, Copilot
  CLI logs.
- `judge_results_full/results_{mode}_{run}_{dataset}_judge{1,2,3}.csv` —
  per-cell scores (one CSV per judge × run × cell).
