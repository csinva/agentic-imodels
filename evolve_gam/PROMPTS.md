# Key prompts

The model in this folder was produced by an autonomous research loop driven by the
prompts below, in order. They are quoted as written; most of the work happened
between them, with the loop proposing ideas, benchmarking them, and keeping or
discarding each on measured evidence.

What is worth noticing is how little of the method was specified. No prompt
mentions Gaussian processes, binning, sufficient statistics, or marginal
likelihood — those came out of the search. The prompts set *objectives*
(beat EBM, stay pairwise, be simple) and repeatedly refused to accept
complexity.

## Setting the problem

> Read and follow the instructions in `program.md`. Focus on building a better GAM
> algorithm. It's okay if it's slower to train, as long as it maintains interpretability.

> Ignore some of the instructions and iterate on GAM algorithms. You don't need to
> rerun the interpertability tests, just make sure that your algorithm is a GAM with
> at most pairwise interations. Do several iterations to get the best predictive performance.

> You must beat the predictive performance of EBM

## Pushing on elegance

> Continue looking for more GAM algorithms, trying to maximize performance but make the
> method simple and elegant rather than several complex things stacked together. Try many
> iterations. [...] Find me something elegant that works!

> Look for a genuine breakthrough. Report performance metrics beyond just mean rank also.

> Try some different radical ideas, thinking from different directions and unifying
> different fields. Try for a simple, elegant model that can dramatically outperform other
> GAM algorithms (while still having at most pairwise interactions). It doesn't have to be
> related to GP at all. It can be using high-dimensional geometry, physics ideas, whatever.

## Forcing it to scale

> Next, try benchmarking our best model vs EBM and TabPFN on some larger standard
> benchmark datasets.

> Keep iterating until you have an elegant method that beats EBM both at small and large scale.

> I want a full win against EBM

## Forcing it to simplify

This is where most of the final method came from — thirteen rounds of ablation,
each removing anything that could not prove its worth.

> Is there any needless complexity in the model that can be removed? Try to make it as
> simple as possible while preserving this performance

> Try another round of simplification to see what else can be removed / must stay.

> Try to make it even simpler again, with replacements that can simplify things.
> The current model is too complicated.

(The last prompt was repeated several times. Each repetition triggered another
ablation round; the model went from 1,110 lines with a bolted-on gradient-boosted
tree ensemble to 686 lines and a single class.)

## Independent validation and write-up

> Okay now identify another popular suite of distinct regression datasets to evaluate on.
> Download and evaluate the newly developed method against EBM and other baselines on this
> new suite.

> Now write a clean interactive report on how this model works. Include the performance
> table (across all three dataset benchmarks and how they were collected/processed).

> Create a pull request at https://github.com/csinva/imodels/ that integrates in this model,
> including adding a section "under our favorite models" and a page on the online docs like
> FIGS has.
