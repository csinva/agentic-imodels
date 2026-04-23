Previously, you ran the evaluation in 01_blade_with_tools.md, that used tools from `../result_libs/apr9-claude-effort=medium-main-result/interpretable_regressors_lib` that were rewritten into `blade-evaluation/interp_models.py` for the custom setting.

Now, I want you to rerun the experiments using a different custom setting. I want you to use the library at `../result_libs_processed/agentic-imodels`, starting by reading the `../result_libs_processed/agentic-imodels/SKILL.md` file. There is no need to postprocess this directory into a single python file, it is already structured enough.

### Changes to make

Look at the `blade-evaluation` folder.

- First, you should delete the logs, custom runs, and interp_models.py results from before.
- Second, you should edit the `prepare.py` prompt for the custom run in `AGENTS_MD_CUSTOM_V2` to point it to the agentic-imodels SKILL.md file

### Running the experiments

- Once the changes are made, run the codex experiments.
- Next, make the changes above to the `blade-evaluation-claude` folder (use the exact same prompts) and then rerun the custom experiments there.

Finally update the results in `../paper-imodels-agentic/content.tex`
