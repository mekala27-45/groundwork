# Chunking strategies

groundwork indexes every document with two chunking strategies at once,
side by side in the same workspace, so the retrieval evaluation harness
can compare them against the same golden questions rather than asking a
reader to trust a claim about which one is better. This document explains
how each one works and what was actually measured. For the full
per-category numbers, see RESULTS.md; this page only pulls out the
comparison that isolates chunking strategy as the one variable.

## Naive: fixed token windows

`groundwork_chunk.naive.chunk_naive` packs the whole document into
256-token windows with 32 tokens of overlap between consecutive chunks
(`DEFAULT_CHUNK_SIZE_TOKENS`, `DEFAULT_OVERLAP_TOKENS`), the same shape
`chunk_size=1000`-style defaults take in most retrieval libraries, built
by hand here rather than imported so the comparison is against a real,
reasonably implemented baseline instead of a strawman. Token counting
goes through `groundwork_chunk.tokenize`: tiktoken's `cl100k_base`
encoding if reachable (itself a network fetch on first use, not a
bundled file), a deterministic approximate counter if not. This strategy
has no idea where a section starts or ends; a chunk boundary lands
wherever the token budget runs out, which means it can, and does, split
a sentence or a paragraph in half across two chunks.

## Structure aware: split on headers first

`groundwork_chunk.structure.chunk_structure` finds section boundaries
before packing anything. A span counts as a header if its font size is
large relative to the document's own median font size, or if it is bold
at a smaller but still above-median size (`_looks_like_header` in
`segments.py`), with a length cap so an unusually long line of body text
at exactly the median size is never mistaken for one. Every detected
section is then packed independently with the identical
`pack_atom_range` function naive chunking uses, at the same 256-token,
32-token-overlap budget, so the only variable between the two strategies
is how atoms are grouped before packing, not a second, differently tuned
packer. Each resulting chunk carries its section's title (`section_title`,
capped at twelve words), which naive chunking's chunks never have. A
document with no detected headers at all falls back to one section
covering the whole body, never an empty result.

## One guarantee both strategies share: a table is never split

`packing.py`'s `_separator` function is the single place that decides how
two adjacent atoms join into text, a blank line on either side of a
table so it reads as its own block, a plain space otherwise, and both the
token-budget check and the final text assembly resolve through that same
function. They did not always: the budget check once joined candidate
atoms with a bare space-joined string while the final text used the
blank-line rule, so a chunk sitting right at the token budget with a
table on its boundary could pass the check and then come out over budget
once actually resolved. A Hypothesis property test
(`test_no_chunk_exceeds_the_token_budget_unless_it_is_a_single_atom`)
found this for real during the build; sharing one join function between
the check and the resolution is what makes the class of bug structurally
impossible to reintroduce, not merely a convention to remember.

## What was actually measured

The `boundary` category in `evalset/questions.yaml` exists specifically
to test this: every question in it needs information that a naive fixed
window is likely to split across two chunks, verified empirically against
a real `search_chunks()` call while the questions were written, not
assumed from reading the source documents by eye. Pulled from RESULTS.md's
own chunking comparison table:

| Workspace | Config | Recall@5 | MRR |
|---|---|---|---|
| ML Evaluation Reference Guide | naive, no rerank | 0.417 | 0.167 |
| ML Evaluation Reference Guide | structure, no rerank | 0.750 | 0.456 |
| ML Evaluation Reference Guide | structure, rerank | 1.000 | 0.806 |
| Meridian Coaching | naive, no rerank | 0.500 | 0.250 |
| Meridian Coaching | structure, rerank | 1.000 | 1.000 |

On the category the strategy was specifically designed to help, structure
aware chunking wins outright in every configuration measured, and the
gap is largest exactly where naive chunking has no reranking to recover
with. That is not the whole story, and RESULTS.md does not let it be:
look at Meridian Coaching's `direct` category in the full table there,
where naive chunking with reranking (1.000 recall@5, 0.917 MRR) beats
structure aware chunking with reranking (0.667, 0.667) on the same
workspace. A document with clean, informative headers rewards structure
aware chunking on questions that span a boundary; a short, FAQ-shaped
document with weak header structure can end up with structure aware
chunks that are worse groupings than naive's fixed windows happen to
produce. Neither strategy dominates the other across every category
measured here, which is why groundwork indexes both and lets the
evaluation harness say so, rather than picking one and asserting it was
the right choice.

## Practical takeaway

Reranking closes most of the gap between the two strategies in every
workspace measured (see RESULTS.md's full retrieval metrics table), which
matters more in practice than the choice of chunking strategy alone: a
naive-chunked, reranked configuration is competitive with, and sometimes
better than, a structure-aware, unreranked one. Structure aware chunking
earns its complexity specifically on boundary-spanning questions against
well-headered source documents; for a corpus where that is not the
dominant question shape, the naive strategy's simplicity is a reasonable
default, not merely the naive one.
