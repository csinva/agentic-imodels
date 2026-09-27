# Key prompts

The solver in this folder was produced by the prompts below, in order, quoted as written.
Most of the work happened between them, with the loop proposing ideas, benchmarking them, and
keeping or discarding each on measured evidence.

## Setting the problem (to the orchestrating agent)

> In ~/tabular, clone https://github.com/csinva/imodels and https://github.com/csinva/agentic-imodels.
> We are going to do some research on sparse linear integer models. Read carefully the recent blog
> posts there on https://csinva.io/imodels/fastsmalltree.html and https://csinva.io/imodels/gpgam.html
> and how they were implemented in the agentic-imodels repo. Your job is to do a full autoresearch
> experiment and write a blog post on it following carefully the style and methods that were used in
> these other two repos. Your goal is to optimize a sparse linear integer model for both performance
> and accuracy. Start with a lit survey, implement baselines, pick a good set of benchmark tasks based
> on their repos' then set up your autoresearch loop to improve them. You can push directly to the
> agentic-imodels repo (make a new folder evolve_slim and put your work in there). You should make a
> new PR into imodels for this.

The orchestrating agent wrote the literature survey (`LITERATURE.md`), the suites, the harness,
the baselines and `program.md`, then started the loop sessions below.

## The loop sessions

Two sessions ran in parallel, each in its own run folder, each started with the same
instructions (`program.md`) and one extra paragraph of focus.

`runs/sep26-run1`:

> Read and follow the instructions in program.md there exactly, starting with the baseline run
> of slim.py as is. Focus for this run: find points with a clearly lower training loss
> (mean_regret) than FasterRisk, while keeping the solver at least as fast as the starting
> solver. Think about what the harness actually scores (it refits the multiplier and intercept
> for your points) and about searching the integer lattice directly. [...] Keep a file notes.md
> in the run folder: one short entry per attempt [...] Do at least 40 iterations.

`runs/sep26-run2`:

> Read and follow the instructions in program.md there exactly, starting with the baseline run
> of slim.py as is. Focus for this run: make the solver much faster than FasterRisk
> (geo_mean_time) without raising the training loss (mean_regret), and then use the saved time
> to lower the loss. Profile where the starting solver spends its time before optimizing. [...]
> Do at least 40 iterations.

## Second round (September 27): new baselines, then loops aimed at a breakthrough

> Search, implement, and test more baselines

(Eight baselines were added: `src/baselines.py`, `LITERATURE.md`.)

> Think harder and do more autoresearch loops. Try to get through a real breakthrough

The orchestrating agent first measured where headroom was left. On the decile suite the shipped
solver is within 0.0005 of the real-valued model's loss on the hidden set and optimal on 49 of 50
enumerable problems, so more loops there could gain little. Two new tracks were built instead
(`tracks/fine`, `tracks/exact`), each run by two loop sessions with the track's `program.md` and
one paragraph of focus:

- `runs/sep27-exact1`: "think hard about the mathematics of VALID lower bounds that prune whole
  FAMILIES of supports [...] so certification reaches k = 5, 7, 10 on real datasets. Before coding a
  bound, write its proof in notes.md; if you cannot prove it, do not use it."
- `runs/sep27-exact2`: "replace the 11^k enumeration on each surviving support with an exact branch
  and bound over the integer points [...] make the whole certification fast [...] then any valid
  pruning of support families you can prove."
- `runs/sep27-fine1`: "a clearly lower training loss by exploiting the threshold structure: detect
  nested chains of columns from the data, add threshold moves [...] Think about radical alternatives
  too, not only tweaks."
- `runs/sep27-fine2`: "first profile where time goes at this width and make the solver scale [...]
  then spend the saved time on a much stronger search [...] to lower the training loss clearly."
