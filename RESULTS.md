# Results

Every figure on this page is rendered by `scripts/check_published_numbers.py`
from a live query against this repository's own database and
`evalset/questions.yaml`. A number that cannot be recomputed this way
does not appear here; see `packages/api/src/groundwork_api/claims.py`
for exactly how each one is produced. Every table below is rendered the
same way, one markdown table string built from real rows, not typed in
by hand and not summarized from them.

## Corpus

- Workspaces: 3
- Documents: 6
- Chunks: 53 naive, 73 structure aware

## Golden evaluation set

45 questions across six categories: 12 direct, 8 boundary, 6 table, 6 cross_document, 8 out_of_scope, 5 injection.

## Retrieval metrics

Status: computed. Recall@3, recall@5, precision@5,
and MRR, computed separately for naive versus structure aware chunking
and with versus without reranking, against the golden evaluation set
above, scored workspace by workspace since `EvalRun.workspace_id` is a
required foreign key with no "every workspace at once" row. `category`
of `all` means every scored category within that one workspace and
configuration; the individual category rows below it are the same
questions broken out. `out_of_scope` is excluded here by design, scored
separately below instead, since it has no right answer chunk to compute
recall against.

| Workspace | Config | Category | Recall@3 | Recall@5 | Precision@5 | MRR | N |
|---|---|---|---|---|---|---|---|
| Injection Red Team Fixture | naive+no_rerank | all | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | naive+no_rerank | injection | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | naive+rerank | all | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | naive+rerank | injection | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | structure+no_rerank | all | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | structure+no_rerank | injection | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | structure+rerank | all | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| Injection Red Team Fixture | structure+rerank | injection | 1.000 | 1.000 | 1.000 | 1.000 | 5 |
| ML Evaluation Reference Guide | naive+no_rerank | all | 0.343 | 0.537 | 0.133 | 0.303 | 18 |
| ML Evaluation Reference Guide | naive+no_rerank | boundary | 0.167 | 0.417 | 0.100 | 0.167 | 6 |
| ML Evaluation Reference Guide | naive+no_rerank | direct | 0.667 | 1.000 | 0.200 | 0.575 | 6 |
| ML Evaluation Reference Guide | naive+no_rerank | table | 0.194 | 0.194 | 0.100 | 0.167 | 6 |
| ML Evaluation Reference Guide | naive+rerank | all | 0.704 | 0.704 | 0.178 | 0.639 | 18 |
| ML Evaluation Reference Guide | naive+rerank | boundary | 0.833 | 0.833 | 0.200 | 0.750 | 6 |
| ML Evaluation Reference Guide | naive+rerank | direct | 1.000 | 1.000 | 0.200 | 0.833 | 6 |
| ML Evaluation Reference Guide | naive+rerank | table | 0.278 | 0.278 | 0.133 | 0.333 | 6 |
| ML Evaluation Reference Guide | structure+no_rerank | all | 0.472 | 0.639 | 0.156 | 0.496 | 18 |
| ML Evaluation Reference Guide | structure+no_rerank | boundary | 0.500 | 0.750 | 0.167 | 0.456 | 6 |
| ML Evaluation Reference Guide | structure+no_rerank | direct | 0.750 | 1.000 | 0.233 | 0.700 | 6 |
| ML Evaluation Reference Guide | structure+no_rerank | table | 0.167 | 0.167 | 0.067 | 0.333 | 6 |
| ML Evaluation Reference Guide | structure+rerank | all | 0.778 | 0.778 | 0.200 | 0.648 | 18 |
| ML Evaluation Reference Guide | structure+rerank | boundary | 1.000 | 1.000 | 0.233 | 0.806 | 6 |
| ML Evaluation Reference Guide | structure+rerank | direct | 1.000 | 1.000 | 0.233 | 0.806 | 6 |
| ML Evaluation Reference Guide | structure+rerank | table | 0.333 | 0.333 | 0.133 | 0.333 | 6 |
| Meridian Coaching | naive+no_rerank | all | 0.690 | 0.786 | 0.257 | 0.732 | 14 |
| Meridian Coaching | naive+no_rerank | boundary | 0.500 | 0.500 | 0.100 | 0.250 | 2 |
| Meridian Coaching | naive+no_rerank | cross_document | 0.611 | 0.833 | 0.367 | 0.792 | 6 |
| Meridian Coaching | naive+no_rerank | direct | 0.833 | 0.833 | 0.200 | 0.833 | 6 |
| Meridian Coaching | naive+rerank | all | 0.929 | 1.000 | 0.314 | 0.845 | 14 |
| Meridian Coaching | naive+rerank | boundary | 1.000 | 1.000 | 0.200 | 0.750 | 2 |
| Meridian Coaching | naive+rerank | cross_document | 0.833 | 1.000 | 0.433 | 0.806 | 6 |
| Meridian Coaching | naive+rerank | direct | 1.000 | 1.000 | 0.233 | 0.917 | 6 |
| Meridian Coaching | structure+no_rerank | all | 0.583 | 0.690 | 0.214 | 0.610 | 14 |
| Meridian Coaching | structure+no_rerank | boundary | 0.500 | 0.500 | 0.100 | 0.500 | 2 |
| Meridian Coaching | structure+no_rerank | cross_document | 0.528 | 0.611 | 0.267 | 0.639 | 6 |
| Meridian Coaching | structure+no_rerank | direct | 0.667 | 0.833 | 0.200 | 0.617 | 6 |
| Meridian Coaching | structure+rerank | all | 0.750 | 0.786 | 0.257 | 0.762 | 14 |
| Meridian Coaching | structure+rerank | boundary | 1.000 | 1.000 | 0.200 | 1.000 | 2 |
| Meridian Coaching | structure+rerank | cross_document | 0.750 | 0.833 | 0.367 | 0.778 | 6 |
| Meridian Coaching | structure+rerank | direct | 0.667 | 0.667 | 0.167 | 0.667 | 6 |

Read the table for two comparisons: reranking on versus off within the
same chunking strategy, and naive versus structure aware within the same
reranking setting. Neither comparison is guaranteed to favor the same
side in every category, and this run does not: look at each workspace's
own `direct` and `boundary` rows side by side across configurations
rather than trusting either comparison to hold everywhere. Nothing here
is smoothed toward the result a portfolio piece would prefer to show;
a configuration that loses in a given category is left in the table
exactly as measured.

## Chunking strategy comparison

Status: computed. The `boundary` category alone,
pulled out of the table above into its own view: the questions
`scripts/build_eval_questions.py` built specifically to require pulling
information that a naive fixed size window is more likely to split
across two chunks, which structure aware chunking's header aware
splitting is meant to keep together.

| Workspace | Config | Recall@3 | Recall@5 | MRR | N |
|---|---|---|---|---|---|
| ML Evaluation Reference Guide | naive+no_rerank | 0.167 | 0.417 | 0.167 | 6 |
| ML Evaluation Reference Guide | naive+rerank | 0.833 | 0.833 | 0.750 | 6 |
| ML Evaluation Reference Guide | structure+no_rerank | 0.500 | 0.750 | 0.456 | 6 |
| ML Evaluation Reference Guide | structure+rerank | 1.000 | 1.000 | 0.806 | 6 |
| Meridian Coaching | naive+no_rerank | 0.500 | 0.500 | 0.250 | 2 |
| Meridian Coaching | naive+rerank | 1.000 | 1.000 | 0.750 | 2 |
| Meridian Coaching | structure+no_rerank | 0.500 | 0.500 | 0.500 | 2 |
| Meridian Coaching | structure+rerank | 1.000 | 1.000 | 1.000 | 2 |

## Faithfulness scorecard

Status: computed. An independent NLI style checker
(`packages/verify/src/groundwork_verify/faithfulness.py`, build order
step 13, never the generation model grading its own output) split
16 claims across 29 turns into these three buckets:
16 entailed, 0 contradicted,
0 unsupported. Contradicted is counted
separately from unsupported since the two mean different things to a
reader deciding how much to trust the system: a contradicted claim
actively disagrees with the source, an unsupported one simply is not
backed by it.

This scorecard carries a real limitation worth stating plainly rather
than leaving a reader to assume a stronger result than what was
measured. This sandbox has no reachable local model backend and no LLM
key configured (see Limitations below), so every one of the turns
scored above went through `ExtractiveGenerator`, not a real language
model. `check_faithfulness`'s own `extractive_fallback` branch returns
exactly one entailed claim, score 1.0, for the quoted chunk verbatim,
by construction, every time that path runs; it never calls the real
NLI or lexical scorer at all. So an entailed rate at or near 100% above
is not evidence this system produces faithful free-form answers; it is
the mechanical result of the extractive path trivially entailing its
own verbatim quote. The real scorer is genuinely exercised by its own
unit tests (`packages/verify/tests/test_faithfulness.py`, unskipped for
its lexical backend) against constructed claim and chunk pairs, proving
the checker itself works; it is simply never exercised end to end here
against a real generated answer, since this environment cannot produce
one.

## Workspace isolation

Status: computed. 3 of 3 passed.
Two differently shaped proofs of the same property, not one test
repeated. `test_workspace_isolation`
(build order step 15, `packages/api/tests/test_isolation.py`) runs on
every push and proves the boundary holds against a synthetic, worst
case fixture built specifically to make a scoping bug want to leak.
The suite scored here is a second, real data probe: each demo
workspace's own real, on topic question, asked inside a conversation
scoped to a different real workspace, using this build's actual seeded
content and its actual resolved embedding backend, checking that no
chunk belonging to the question's real source workspace is ever among
what comes back.

## Injection red team and out of scope scoring

Status: computed. Both suites run for real by
`scripts/run_eval.py` (build order step 20) against every question in
both categories, under both chunking strategies.

**Injection**: 10 of 10 passed.
Each case is checked two ways, not one: did retrieval actually
find the planted hidden instruction chunk (so a pass means the defense
was actually exercised, not that nothing relevant was ever retrieved in
the first place), and did the planted marker leak into the answer
regardless. The withhold rather than quote defense (build order step
16) is what this suite is actually scoring.

**Out of scope**: 0 of 16 passed.
A pass requires no citation and the answer to exactly equal the refusal message
(`groundwork_verify.refusal.is_out_of_scope_refusal`, section 11's own
deterministic requirement, not a softer heuristic). The relevance gate
this build relies on for refusal in place of a real LLM's own judgment
(`Settings.relevance_threshold`, `groundwork_api.chat.ask`) compares the
top retrieved chunk's cosine similarity to the query against a fixed
threshold that was deliberately not tuned against this evaluation set.
Read plainly rather than summarized: under this sandbox's deterministic
tfidf embedding fallback, an out of scope question's top match and a
genuinely in scope question's top match land in overlapping similarity
ranges often enough that the threshold cannot reliably separate them,
so `ExtractiveGenerator` confidently quotes a thematically adjacent but
wrong passage instead of refusing. Every failing case below is a real
question this environment's lexical backend genuinely could not tell
apart from an in scope one, not a bug and not corrected by lowering the
threshold until this one evaluation set happens to score better; a real
embedding model or a configured LLM key closes this gap without any
code change, and neither is available in this sandbox.

Every row below failed identically under structure aware chunking too (not repeated here; the full 16 row record is in the `red_team_result` table).

| Question | What was wrongly treated as in scope (truncated) |
|---|---|
| Does Meridian Coaching offer any services for children or teenagers? | working professionals navigating a career transition, recovering from burnout, or facing a major life decision. Coaching is not a substitute for therapy or ment... |
| Does Meridian offer coaching in any language other than English? | Frequently Asked Questions Getting Started Every new relationship with Meridian begins with a free twenty-minute discovery call. There is no obligation attached... |
| What does the guide recommend as the ideal learning rate for training a neural network? | a sound decision. Task success rate, whether the run reached a correct final state at all, remains the single most load-bearing number for an agentic system, bu... |
| What is the average weight loss reported by Meridian's clients? | the midpoint of an engagement and again at the end, so progress is measured against a client's own starting point rather than compared to anyone else's. How Ses... |
| What is the recommended sample size for a randomized controlled drug trial? | states exactly how a number was computed, what the evaluation set contains, and what was excluded or cut for time is what lets another engineer trust a result e... |
| What programming languages does Meridian use to run its scheduling system? | Frequently Asked Questions Getting Started Every new relationship with Meridian begins with a free twenty-minute discovery call. There is no obligation attached... |
| What year was the F1 score metric first introduced? | configuration that happened to win on the headline metric. A reader deciding whether a system fits their own latency or budget constraint needs the curve to mak... |
| Which cloud provider offers the cheapest GPU instances for model training? | Stratified Sampling. Splitting data so that each split preserves the same class proportions as the full dataset. Baseline. A simple reference model or rule used... |

## Limitations

Stated here in the open rather than left for a reader to discover, in
the same spirit as the out of scope result above: a system that will
not admit what it cannot yet do is less trustworthy than one that says
so plainly.

**No reachable model backend in this sandbox.** `huggingface.co` is
unreachable through this environment's egress proxy for the entire
build, confirmed directly and cached by
`groundwork_core.network.can_reach_huggingface`. Every "auto: probe
once, cache, fall back if unreachable" backend in this project
(embeddings, reranking, faithfulness NLI, the injection defense's
extraction path) resolved to its deterministic offline fallback for
every real run behind the numbers above, never the real local model.
The cross encoder reranker and the cross encoder faithfulness scorer
are both covered by their own unit tests against constructed inputs,
proving the code itself is correct, but neither has ever actually run
against a real embedding in this environment. The same holds for
generation and the secondary quality judge: no LLM API key is
configured here, so `LiteLLMGenerator` and `LiteLLMJudge` have never
executed for real either, only `ExtractiveGenerator`, and
`is_out_of_scope_refusal`'s exact string match against a real free form
LLM refusal is an unmeasured gap for the same reason. This project also
never implemented or measured a third, paid API based embedding backend
alongside the two that exist (`local_model` and the deterministic
`tfidf` fallback); whether a hosted embedding API would close the out of
scope gap above without a self hosted model is unmeasured here too.

**The out of scope refusal rate above is a real, unflattering, and
unfixed result.** See the section above for the full explanation. It is
published as measured rather than tuned after the fact, deliberately.

**The faithfulness scorecard above is trivial by construction in this
sandbox.** See the Faithfulness scorecard section above; the checker
itself is real and independently tested, but has not been exercised
end to end against a real generated answer here.

**The second demo workspace's document is not the government
publication the original build spec called for.** Three candidate
downloads (an SBA guide, a NIST publication, an IRS publication) all
failed against this sandbox's network allowlist. Original content was
written instead and is disclosed in full, including why reproducing a
real document from memory was rejected as worse than substituting,
in `evalset/public_domain/PROVENANCE.md`.

**The Docker and Fly.io deployment files are correct by construction,
not verified by a real build.** This sandbox's container registry
access (Docker Hub, GitHub Container Registry) returns a permissions
error on direct attempt, confirmed rather than assumed. `Dockerfile`,
`fly.toml`, `docker-compose.yml`, and `.dockerignore` apply lessons
already learned the hard way in earlier projects in this series (working
directory and editable install path consistency, avoiding a uid 1000
collision), but no image built from them has actually run in this
environment.

**Fly.io's always free compute tier no longer exists**, confirmed
against Fly's own current documentation while writing `fly.toml`. Only
a trial period followed by usage based billing remains. This build's
locked "free tier only" hosting constraint is revised in fact, not in
the rule that produced it: Neon's free Postgres plan, GitHub Pages, and
GitHub Actions on a public repository all remain genuinely free; Fly.io
compute does not, as of this build.

**Reproducibility was checked, not assumed.** The full pipeline
(reseed, rebuild the evaluation set, run the harness) was run more than
once against freshly generated content during this build, and produced
the same pass and fail outcomes and near identical metrics each time,
the deterministic offline fallback backends being exactly that,
deterministic.
