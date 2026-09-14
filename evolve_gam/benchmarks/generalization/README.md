# Generalization harness (Sep 2026 iterations, v48 and v49)

Multi-seed evaluation on the imodels-65 development suite and the de-duplicated
held-out sets. Set `GEN_DIR` to a scratch directory (it will hold `data/`, `data_large/`,
`results/`), run `cache_data.py` once, then e.g.

    uv run run.py v49 --model=gpm.BinGP --seeds=0,1,2 --jobs=3 "schedule='v49h'" tau=1.0 learn_scales=True scale_prior=0.5
    uv run report.py v48 v49          # paired comparison with the pre-registered acceptance rule
    uv run heldout_eval.py v49h       # TabArena-12 + CTR23-23 (de-duplicated), identical preprocessing for all models
    uv run heldout_report.py

`gpm.BinGP` refers to the research workbench (`../../model/addgp.py` exposes the shipped
model; the workbench with every falsified flag is not shipped). Seed 0 is the official
harness split; seeds > 0 redraw the split from pooled train+test.
