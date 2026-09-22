# Results

Every figure on this page is rendered by `scripts/check_published_numbers.py`
from a live query against this repository's own database and
`evalset/questions.yaml`. A number that cannot be recomputed this way
does not appear here; see `packages/api/src/groundwork_api/claims.py`
for exactly how each one is produced.

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
above. The harness itself is build order step 11; `scripts/run_eval.py`
(build order step 20) is what actually runs it and persists an `EvalRun`
row per workspace, configuration, and category. The full comparison
table renders here at build order step 21, the commit immediately after
this status first reads "computed".

## Chunking strategy comparison

Status: computed. Naive versus structure aware
chunking, compared on the same metrics as the retrieval table above,
with the boundary spanning question category broken out separately and
reported honestly even where naive chunking wins or ties. Renders here
alongside the retrieval metrics table at build order step 21.

## Faithfulness scorecard

Status: computed. Entailed, contradicted, and unsupported
claim rates from the independent NLI faithfulness checker (build order
step 13), contradicted counted separately from unsupported since the two
mean different things to a reader deciding how much to trust the system.
Renders here at build order step 21.

## Workspace isolation

Status: computed. `test_workspace_isolation` (build order
step 15), a synthetic worst case fixture built to make a scoping bug
want to leak, plus a second, differently shaped probe against this
build's real seeded content, both run for real by `scripts/run_eval.py`
(build order step 20). Renders here at build order step 21.

## Injection red team and out of scope scoring

Status: computed. The injection defense test and the out of
scope refusal scoring (build order steps 16 and 17), run for real by
`scripts/run_eval.py` (build order step 20) against every question in
both categories, under both chunking strategies. Renders here at build
order step 21.

## Limitations

This section is filled in at build order step 21, once every result
above is in hand, and names what was cut or substituted along the way,
including `evalset/public_domain/PROVENANCE.md`'s documented departure
from the original public domain document requirement.
