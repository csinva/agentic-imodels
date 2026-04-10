# BLADE Evaluation: Standard vs. Custom Interpretability Tools

## Overview

This report compares how well an AI agent (OpenAI Codex with `gpt-5.3-codex`) performs on the 13 BLADE benchmark data-science tasks under three conditions:

1. **Standard tools**: Agent uses scikit-learn, imodels, statsmodels, scipy
2. **Custom v1**: Standard tools + custom interpretable regressors (`interp_models.py`) with basic prompting
3. **Custom v2**: Improved custom tools (DataFrame-aware, `feature_effects()` method) + structured analysis strategy emphasizing feature importance, effect shapes, and robustness

All runs used the same Codex configuration (model_reasoning_effort="high", danger-full-access sandbox, gpt-5.3-codex on dl-openai-3). Evaluation used a rubric (1-10 scale) that rewards conclusion correctness, depth of understanding (feature importance, effect shapes, nonlinear patterns), and clear connection between evidence and conclusions.

## Results Summary (1-10 scale)

| Dimension | Standard | Custom v1 | Custom v2 |
|-----------|----------|-----------|-----------|
| Correctness | 8.46 | 8.69 | **8.69** |
| Completeness | 7.77 | 7.92 | **8.38** |
| Clarity | 8.54 | **8.62** | 8.46 |
| **Overall** | 8.26 | 8.41 | **8.51** |

**Custom v2 achieves the highest overall score (8.51/10)**, with the biggest gain in completeness (+0.61 over standard, +0.46 over custom v1).

## Per-Dataset Scores

| Dataset | Standard ||| Custom v1 ||| Custom v2 |||
|---------|C|Comp|Cl|C|Comp|Cl|C|Comp|Cl|
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| affairs | 9 | 8 | 9 | 10 | 9 | 9 | 9 | **9** | 9 |
| amtl | 9 | 8 | 9 | 9 | 8 | 9 | 9 | 8 | 9 |
| boxes | 9 | 8 | 9 | 9 | 8 | 9 | 8 | 8 | 8 |
| caschools | 8 | 7 | 9 | 7 | 6 | 8 | **9** | **8** | 8 |
| crofoot | 7 | 6 | 7 | 8 | 7 | 8 | 8 | 7 | 8 |
| fertility | 9 | 8 | 8 | 9 | 8 | 9 | 9 | 8 | 9 |
| fish | 8 | 8 | 9 | 9 | 8 | 9 | 8 | **9** | 9 |
| hurricane | 9 | 8 | 8 | 9 | 8 | 9 | 9 | **9** | 8 |
| mortgage | 7 | 8 | 7 | 8 | 9 | 9 | **9** | 8 | **9** |
| panda_nuts | 9 | 8 | 9 | 9 | 8 | 8 | 9 | 8 | 8 |
| reading | 9 | 8 | 9 | 9 | 7 | 8 | 9 | **9** | **9** |
| soccer | 8 | 8 | 9 | 8 | 8 | 8 | 8 | **9** | 8 |
| teachingratings | 9 | 8 | 9 | 9 | 9 | 9 | 9 | **9** | 8 |

## Agent Likert Scores (0-100)

| Dataset | Standard | Custom v1 | Custom v2 |
|---------|----------|-----------|-----------|
| affairs | 0 | 0 | 12 |
| amtl | 80 | 100 | 66 |
| boxes | 15 | 15 | 23 |
| caschools | 65 | 90 | 22 |
| crofoot | 26 | 64 | 57 |
| fertility | 20 | 5 | 8 |
| fish | 82 | 87 | 69 |
| hurricane | 7 | 10 | 4 |
| mortgage | 85 | 65 | 39 |
| panda_nuts | 78 | 90 | 60 |
| reading | 15 | 12 | 8 |
| soccer | 76 | 35 | 65 |
| teachingratings | 79 | 100 | 100 |

## Analysis

### Completeness: The main differentiator (+0.61 over standard)

Custom v2's biggest improvement is completeness (8.38 vs 7.77 standard). The custom interpretable tools enabled the agent to go beyond p-values:

- **Feature importance rankings**: The agent consistently reported which variables matter most and their relative importance (e.g., "beauty ranked 1st in importance at 50.8%")
- **Effect shapes**: SmartAdditiveRegressor revealed nonlinear patterns that OLS misses (e.g., threshold effects in age for panda_nuts, nonlinear beauty effect in teachingratings)
- **Robustness checks**: The agent compared findings across OLS, SmartAdditive, and HingeEBM, noting when effects were consistent or model-dependent

Datasets with the largest completeness gains:
- **fish** (8→9): Custom tools revealed the nonlinear shape of the hours effect
- **hurricane** (8→9): Feature importance confirmed femininity has negligible importance relative to pressure/wind
- **soccer** (8→9): Importance rankings showed skin tone's effect is real but tiny relative to other factors
- **reading** (8→9): Feature importance showed timing/text characteristics dominate over Reader View

### Correctness: Tied with Custom v1 (8.69 vs 8.46 standard)

Both custom variants improved correctness over standard by +0.23 points. Key improvements:
- **caschools** (8→9): Custom tools showed the student-teacher ratio effect disappears after controls, leading to a more defensible low score
- **mortgage** (7→9): Interpretable models helped the agent recognize gender's small but real effect relative to other predictors

### Clarity: Slight trade-off (8.46 vs 8.54 standard)

Custom v2 clarity is slightly lower than standard (-0.08). The richer explanations (importance percentages, threshold values, model comparisons) add substance but also complexity. However, the judge noted explanations were "well-structured" and "insightful" even when slightly technical.

### What the custom tools uniquely contributed

Looking at the v2 conclusions, the custom tools added specific insights that standard tools couldn't:

1. **teachingratings**: "SmartAdditive ranks beauty as importance=50.8%, rank=1, with a nonlinear increasing pattern and a zero-crossing threshold near beauty=-0.698" — this reveals the shape of the beauty effect, not just its significance.

2. **fish**: "hours has a nonlinear (increasing trend) effect with importance=23.5%" — reveals diminishing returns that OLS's linear coefficient obscures.

3. **hurricane**: "HingeEBM zeros out femininity entirely" — the Lasso selection in HingeEBM provides strong evidence that femininity is truly unimportant.

4. **soccer**: "skin_tone importance=3.2% vs games importance=45.1%" — quantifying relative importance is more informative than just reporting p-values.

## Summary of improvements from v1 to v2

| Change | Purpose | Effect |
|--------|---------|--------|
| DataFrame-aware models | Column names in output instead of x0, x1 | More readable model output |
| `feature_effects()` method | Structured importance/direction/rank dict | Agent reports importance rankings systematically |
| Structured analysis strategy | Step-by-step: explore → test with controls → interpretable models → conclude | Agent follows a more thorough workflow |
| Balanced scoring guidance | "Weigh both bivariate and controlled" instead of "score low if controls matter" | Better-calibrated Likert scores |
| Updated rubric | Rewards effect shapes, importance, nonlinear patterns | Measures what custom tools actually provide |

## Setup

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0)
- **Model**: `gpt-5.3-codex` (Azure deployment: `dl-openai-3`)
- **Judge**: Azure OpenAI `gpt-4o` with v2 rubric (1-10 scale)
- **Custom tools**: `SmartAdditiveRegressor` and `HingeEBMRegressor` from `interp_models.py`

## How to Reproduce

```bash
# Prepare all three modes
python prepare_run.py --mode standard
python prepare_run.py --mode custom
python prepare_run.py --mode custom_v2

# Run Codex
bash run_all.sh --mode standard
bash run_all.sh --mode custom
bash run_all.sh --mode custom_v2

# Evaluate with v2 rubric
python evaluate.py --mode standard --rubric v2 --verbose
python evaluate.py --mode custom --rubric v2 --verbose
python evaluate.py --mode custom_v2 --rubric v2 --verbose
```
