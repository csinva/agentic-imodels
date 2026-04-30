# Blade Evaluation — Copilot CLI + Gemini 2.5 Pro

This folder mirrors `blade-evaluation-copilot/` but drives the Copilot CLI with
Gemini 2.5 Pro instead of Claude Sonnet 4.5.

## Setup

- Agent: `~/.local/bin/copilot-gemini -p ... --model gemini-2.5-pro
  --allow-all-tools --allow-all-paths --no-color`,
  hard-capped at `timeout 1200s` per dataset.
- Wrapper: invokes Copilot CLI's JS bundle via Node 24 because the precompiled
  Linux binary at `copilot-linux-x64/copilot` is a Node SEA whose hardcoded
  `l_n` blocklist hides Gemini models. The JS bundle's `app.js` was patched
  locally to remove `gemini-2.5-pro` from that set.
- Note: Gemini 3.1 (Preview) is not yet exposed in the Copilot CLI for this
  account; only `gemini-2.5-pro` is in the server-returned model list, so we
  use that.
- Judge: Azure OpenAI `gpt-4o` via keyless Entra ID auth (unchanged from
  original pipeline).
- Datasets: 13 Blade tasks, sourced from
  `../blade-evaluation-codex/outputs_standard_run1/` (info.json + CSV per
  dataset).
- Modes:
  - `standard` — agent instructed to use scikit-learn / imodels / statsmodels
  - `custom_v2` — `standard` plus the `agentic_imodels` package (10 evolved
    regressors) and `SKILL.md` (API + recommended analysis workflow) copied
    into each run directory.
- Runs: 3 agent runs per mode × 3 judge runs = **9 evaluations per dataset
  per mode**.
- Failed cells were re-run sequentially until each (mode, run, dataset) had a
  conclusion. Total: 78/78 conclusions, 234 judge evaluations.

## Results (mean ± SE across 13 datasets, 1–10 scale, n=9 evaluations per cell)

| Dimension     | standard         | custom_v2        | Δ (custom − std) |
| ------------- | ---------------- | ---------------- | ---------------- |
| Correctness   | 4.79 ± 0.23      | **7.68 ± 0.39**  | **+2.89**        |
| Completeness  | 4.03 ± 0.23      | **7.29 ± 0.30**  | **+3.26**        |
| Clarity       | 4.49 ± 0.27      | **7.90 ± 0.34**  | **+3.41**        |
| **Overall**   | **4.43**         | **7.62**         | **+3.19 (+71.9%)** |

**All 13 / 13 datasets improved** under custom_v2. SE bars do not overlap on
any dimension.

## Per-Dataset Scores (mean ± SE, n=9)

| Dataset         | Std Corr | Std Comp | Std Clar | Cus Corr | Cus Comp | Cus Clar |
|-----------------|----------|----------|----------|----------|----------|----------|
| affairs         | 3.2±0.1  | 2.6±0.2  | 2.8±0.2  | 5.6±0.9  | 5.9±0.7  | 6.2±0.7  |
| amtl            | 5.1±0.2  | 4.0±0.2  | 4.2±0.2  | **8.8±0.1** | **7.9±0.2** | **8.8±0.1** |
| boxes           | 5.1±0.9  | 5.1±0.7  | 5.4±0.9  | 6.6±1.0  | 6.4±0.7  | 7.0±0.8  |
| caschools       | 5.4±0.2  | 4.2±0.2  | 5.2±0.2  | **8.2±0.2** | **7.6±0.3** | **8.3±0.2** |
| crofoot         | 3.6±0.3  | 3.0±0.3  | 3.3±0.3  | **8.9±0.1** | **8.1±0.1** | **8.8±0.1** |
| fertility       | 5.8±0.7  | 4.3±0.8  | 5.3±0.8  | 6.9±0.9  | 6.8±0.6  | 7.4±0.7  |
| fish            | 4.3±0.3  | 4.2±0.2  | 4.0±0.4  | **8.3±0.2** | **8.1±0.2** | **8.8±0.1** |
| hurricane       | 5.6±0.6  | 4.7±0.6  | 5.0±0.4  | **8.8±0.1** | **8.2±0.1** | **8.7±0.2** |
| mortgage        | 4.7±0.2  | 4.2±0.3  | 4.8±0.4  | **8.7±0.2** | **8.1±0.1** | **8.9±0.1** |
| panda_nuts      | 4.9±0.2  | 4.2±0.3  | 4.8±0.2  | **8.9±0.1** | **8.0±0.0** | **8.9±0.1** |
| reading         | 4.4±0.2  | 3.0±0.2  | 3.7±0.2  | **6.1±0.8** | **5.8±0.6** | **6.4±0.6** |
| soccer          | 4.1±0.2  | 3.4±0.4  | 3.7±0.3  | 5.2±0.8  | 5.3±0.6  | 5.4±0.7  |
| teachingratings | 6.0±0.6  | 5.3±0.5  | 6.1±0.5  | **8.9±0.1** | **8.6±0.2** | **9.0±0.0** |

## Comparison to Claude Sonnet 4.5 (`blade-evaluation-copilot/`)

| Run                       | std overall | cus overall | Δ (abs) | Δ (%)   |
| ------------------------- | ----------- | ----------- | ------- | ------- |
| Copilot + Claude 4.5      | 6.16        | 8.15        | +1.99   | +32.3%  |
| Copilot + Gemini 2.5 Pro  | 4.43        | 7.62        | +3.19   | +71.9%  |

Gemini's standard baseline is markedly weaker than Claude's (4.43 vs 6.16),
but custom_v2 reaches a comparable quality level (7.62 vs 8.15). Net effect:
agentic-imodels delivers **nearly twice the relative improvement** under the
weaker base model.

## Reliability notes

- 5 cells timed out at 1200 s in their first attempt (all `custom_v2`):
  `run2/{caschools, fish, soccer}`, `run3/{panda_nuts, soccer}`. In every case
  Gemini 2.5 Pro entered an "I'll read AGENTS.md..." reasoning loop and
  failed to invoke shell tools. Each was re-run sequentially; all but
  `run2/caschools` recovered on the first retry, `run2/caschools` recovered
  on the fourth attempt. No cells were dropped from the analysis.
- A single early run with three datasets (`boxes, crofoot, fish`) produced a
  misleading negative pilot because of one anomalously bad `crofoot`
  custom_v2 conclusion. The 13-dataset pilot and the full sweep both show a
  large positive effect — the 3-dataset signal was noise.

## Files

- `run_all.sh` — runs `copilot-gemini -p` per dataset (timeout 20 min).
- `prepare_run.py` — builds per-dataset run dirs (uses sibling repo's data).
- `evaluate.py` — Azure OpenAI LLM-as-a-judge (unchanged from original).
- `aggregate_results.py` — aggregates judge CSVs (unchanged).
- `outputs_standard_run{1,2,3}/`, `outputs_custom_v2_run{1,2,3}/` — per-dataset
  working dirs with `analysis.py`, `conclusion.txt`, Copilot CLI logs.
- `judge_results_full/results_{mode}_{run}_{dataset}_judge{1,2,3}.csv` — per-cell scores (one CSV per judge × run × cell).
- `judge_results/` — single-pass judge1 results retained for the simpler
  3-eval-per-cell analysis.
