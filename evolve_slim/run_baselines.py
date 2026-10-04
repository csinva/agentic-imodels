"""Run the fixed baselines on a suite and record them. Not modified by the agent.

    uv run run_baselines.py                                  # every baseline, visible suite
    uv run run_baselines.py --models fasterrisk --suite hidden
    uv run run_baselines.py --models fasterrisk --datasets heart,mammo --ks 3 --no-record

Rows go to results/overall_results.csv and results/problem_results.csv (visible),
or results/<suite>_*.csv for another suite.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from baselines import BASELINES, INTEGER  # noqa: E402
from evaluate import evaluate_solver, print_summary, record  # noqa: E402
from suite import RESULTS_DIR, TIME_LIMIT  # noqa: E402

DESCRIPTIONS = {
    "fasterrisk": "FasterRisk 0.1.10 (Liu et al. 2022), default settings: beam search, diverse pool, star-ray rounding",
    "fasterrisk_wide": "FasterRisk with every search width raised (beam 50x50, pool 200, 100 swaps, 100 multipliers)",
    "riskslim": "RiskSLIM lattice cutting planes (Ustun & Rudin 2019), CPLEX 22.1 Community Edition, time limit",
    "slim_milp": "SLIM (Ustun & Rudin 2016), 0-1 loss MILP solved by HiGHS within the time limit",
    "rounded_lr": "L1 logistic regression tuned to k features, refit, scaled to a largest point of 5, rounded",
    "imodels_slim": "imodels 3.0.2 SLIMClassifier without a MIP solver (rounded L2 logistic), penalty searched to k",
    "unit_weighting": "unit weighting: L1-selected features (at most k), each worth +1 or -1 by its sign",
    "autoscore": "AutoScore (Xie et al. 2020) on binary features: RF ranking, LR, coefficients / smallest, rounded",
    "l1path_seqround": "FasterRisk's star-ray sequential rounding on every L1-path support of size <= k",
    "cpa_highs": "RiskSLIM's cutting-plane algorithm with the open-source HiGHS MILP solver (re-solved each round), time limit",
    "abess_seqround": "abess best-subset logistic (sizes 1..k), FasterRisk's star-ray sequential rounding",
    "fastsparse_seqround": "fastSparse L0L2 logistic path (support <= k, box [-5,5]), star-ray sequential rounding",
    "okridge_seqround": "OKRidge optimal k-sparse ridge support, logistic refit, star-ray sequential rounding",
    "psl": "probabilistic scoring list (scikit-psl 0.7.2), scores in +-{1..5}, k greedy stages",
    "riskscores": "riskscores 1.3.0 (R) risk_mod, annealscore, points in [-5,5], lambda0 path + bisection to <= k points",
    "riskscores_cd": "riskscores 1.3.0 (R) risk_mod, riskcd coordinate descent, points in [-5,5], lambda0 path + bisection",
    "skscope_seqround": "skscope 0.1.8 ScopeSolver k-sparse logistic (sizes 1..k), star-ray sequential rounding",
    "l0learn_seqround": "L0Learn 2.1.0 (R) logistic L0L2 path with CDPSI swaps, support <= k, star-ray sequential rounding",
    "okglm_seqround": "OKGLM (ICML 2025) BnB for box k-sparse logistic (CPU, half the limit), star-ray sequential rounding",
    "continuous_beam": "reference, real-valued: FasterRisk's k-sparse beam search before rounding",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(b for b in BASELINES if b != "fasterrisk_wide"))
    ap.add_argument("--suite", default="visible")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--ks", default="")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--tag", default="", help="suffix for the model name (e.g. a repeat index)")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    datasets = [d for d in args.datasets.split(",") if d] or None
    ks = [int(v) for v in args.ks.split(",") if v] or None
    for name in args.models.split(","):
        t0 = time.time()
        label = name + args.tag
        s = evaluate_solver(("baseline", name), label, suite=args.suite, datasets=datasets, ks=ks,
                            time_limit=args.time_limit, jobs=args.jobs, integer=INTEGER[name])
        print_summary(label, s, args.time_limit)
        print(f"total_seconds: {time.time() - t0:.1f}s")
        if not args.no_record and datasets is None and (ks is None or args.suite != "visible"):
            prefix = "" if args.suite == "visible" else f"{args.suite}_"
            if args.time_limit != TIME_LIMIT:
                prefix += f"t{args.time_limit:g}_"
            record(label, DESCRIPTIONS[name], s, status="baseline", results_dir=RESULTS_DIR,
                   integer=INTEGER[name], prefix=prefix)
