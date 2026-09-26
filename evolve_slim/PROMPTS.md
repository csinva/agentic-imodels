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
