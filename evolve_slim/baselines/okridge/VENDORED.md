# vendored from okridge 0.1.1 (PyPI), BSD 3-Clause, see LICENSE. Its metadata pins numpy<2 but the code runs on numpy 2.
# evolve_slim patches: __init__ no longer reads the installed version; node.py finetune_ADMM_rho called the
# removed Node.lower_solve, now lower_solve_fast (lines marked evolve_slim).
