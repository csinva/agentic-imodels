# Background

The goal is to evaluate how well different classes in the `interpretable_regressors_lib` folder perform on (1) various regression tasks and (2) on interpretability metrics. The original results are stored in the `original_results` folder, particularly the `original_results/overall_results.csv` file.

The original results were obtained by running the `../evolve/interpretable_regressor.py` script. That script makes calls to `../evolve/src/performance_eval.py` and `../evolve/src/interp_eval.py` to compute the performance and interpretability metrics, respectively.

# Your task

Your task is to evaluate all of the classes in the `interpretable_regressors_lib` folder on **new** regression tasks and **new** interpretability metrics and save the results in a new folder called `new_results`. To do so, you should create a new script called `evaluate_new_generalization.py` that performs this evaluation. The new regression tasks and interpretability metrics should be different from those used in the original results. Also include all of the original models from the `../evolve/run_baselines.py` script in your evaluation.

To evaluate predictive performance, write a script to use the following OpenML ids to get datasets: [44065, 44066, 44068, 44069, 45048, 45041, 45043, 45047, 45045, 45046, 44055, 44056, 44059, 44061, 44062, 44063]. (These are from suite 335 in <https://github.com/LeoGrin/tabular-benchmark>, 1 dataset is removed for overlapping with training (abalone).)

To evaluate interpretability write a script that makes minor variations of every test in the original interpretability metrics, e.g. use slightly different synthetic inputs.

Run them all and save / visualize the results into the `new_results` folder, following the same format as the original results. Try to import and reuse functions from previous scripts whenever possible rather than replicating code. Use the `original_results/overall_results.csv` file as a template for how to format your new results.
