# Faithfulness and citation verification

An answer that cites a source is not automatically an answer that source
actually supports. groundwork treats those as two separate questions,
checked by two separate mechanisms, plus a third step that turns free
text into something either mechanism can check at all. This document
explains how, and where the distinction between them shows up in a real
result. For the measured numbers this produces, see RESULTS.md's
Faithfulness scorecard section; this page explains the mechanism behind
those numbers.

## Step one: splitting an answer into claims

`groundwork_verify.split.split_claims` turns a generated answer into
sentence-level claims, each resolved to the chunk its `[N]` citation
marker points at. A marker does not have to sit on the sentence it
belongs to: "Sessions run fifty minutes. They are held over video call.
[1]" is one point made across two sentences, so an unmarked sentence
inherits the nearest marker found after it, resolved by scanning the
answer in reverse. A sentence with no marker anywhere after it, or a
marker naming an index outside the range of chunks actually offered
(a model hallucinating a citation number, or simply off by one), resolves
to `cited_chunk_id=None` rather than raising: a malformed citation is
exactly what this verification layer exists to catch and report, not
something to crash on.

## Step two: faithfulness, independent of the model that answered

`groundwork_verify.faithfulness.check_faithfulness` scores whether each
claim's cited chunk actually entails it, using a model that never once
calls back into generation. This is a deliberate conflict of interest
rule, not an implementation detail: a model grading its own output would
share the exact blind spots that produced a hallucination in the first
place, so the faithfulness checker imports nothing from
`groundwork_generate` and nothing from LiteLLM. Two backends behind one
`FaithfulnessScorer` interface, the same auto-probe shape every
model-backed component in this project uses: `CrossEncoderFaithfulnessScorer`
(a local NLI cross encoder, `cross-encoder/nli-deberta-v3-base` by
default) if `huggingface.co` is reachable, `LexicalFaithfulnessScorer` (word
overlap between claim and chunk, a negation mismatch on an otherwise
high-overlap pair standing in for contradiction) if not. The lexical
backend requires 60% of a claim's vocabulary to be traceable to its cited
chunk before calling it supported at all (`ENTAILMENT_OVERLAP_THRESHOLD`);
below that it is unsupported regardless of polarity. Neither backend is
dressed up as equivalent to the other; the lexical one is named and
documented as a real, if less sophisticated, substitute, not a stub.

The extractive fallback (`ExtractiveGenerator`, the only generation path
this sandbox has ever exercised for real, since no LLM key is configured
here) skips the scorer entirely rather than running it pointlessly. Its one claim is the top chunk's text quoted verbatim, by
construction, so it is faithful by construction too: probabilistically
re-checking whether a verbatim quote entails itself would not test
anything. The one exception is a top chunk `groundwork_ingest.security`
flagged as suspicious: `ExtractiveGenerator` withholds that text instead
of quoting it, so there is no claim about the document to score at all,
the same empty result an empty retrieval list already produces.

## Step three: citation verification, a fact check, not a judgment

`groundwork_verify.citation.verify_citations` is deliberately the
simplest piece of this pipeline: no model, no probability, three
deterministic checks against the database for every cited chunk. Does
the chunk still exist. Does it still belong to the conversation's own
workspace, the same boundary `test_workspace_isolation` checks elsewhere.
Does its current stored text still sha256-hash identically to the exact
copy that was actually offered to the generator at answer time. That
last check is a hash comparison specifically, never a substring search,
so a chunk that was edited or re-indexed out from under a citation is
caught even if its new text happens to still contain a matching phrase
somewhere.

## The distinction that matters: verified is not the same as correct

Citation verification answers one narrow question: does the quoted text
genuinely come from the chunk it claims to. It says nothing about
whether that chunk actually answers the question asked. This shows up
directly in this build's own real results: every out-of-scope failure in
RESULTS.md's red team table shows `verified: true`, because
`ExtractiveGenerator` quotes real, unaltered text from a real, existing
chunk every time, exactly what citation verification checks for. The
chunk is simply the wrong chunk for the question, something citation
verification was never designed to catch and faithfulness scoring cannot
catch either, since an extractive answer is faithful to its own citation
by construction regardless of whether that citation was the right one to
retrieve in the first place. Retrieval relevance is a separate concern,
covered by the relevance gate (`Settings.relevance_threshold`) and
measured by the retrieval evaluation harness, not by anything in this
document. The web app's `/trace` view states this distinction directly
next to its citation verification section, rather than letting a green
"verified" checkmark read as "correct."

## What this sandbox could and could not measure

The faithfulness checker's real cross-encoder backend is covered by its
own unit tests (`packages/verify/tests/test_faithfulness.py`) against
constructed claim and chunk pairs built specifically to exercise entailed,
contradicted, and unsupported outcomes, proving the scoring logic itself
is correct. It has never run end to end here against a real generated
answer, because this sandbox has no LLM key and cannot reach
`huggingface.co`, so every real turn behind the published faithfulness
scorecard went through the extractive path's automatic entailed result
instead. RESULTS.md's own Limitations section states this plainly: a
faithfulness scorecard reading close to 100% entailed here is evidence
the extractive fallback works as designed, not evidence about how
faithful this system's free-form, LLM-generated answers would be, since
none were ever produced to measure.
