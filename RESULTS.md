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

Status: not yet run. This section populates with
recall@3, recall@5, precision@5, and MRR, computed separately for naive
versus structure aware chunking, with and without reranking, once the
retrieval evaluation harness (build order step 11) runs against the eval
set above.

## Chunking strategy comparison

Status: not yet run. Populated alongside the retrieval
metrics above, with the boundary spanning question category broken out
separately, reported honestly even where naive chunking wins or ties.

## Faithfulness scorecard

Status: not yet run. Populated once the independent NLI
faithfulness checker (build order step 13) runs, with contradicted
claims counted separately from unsupported ones.

## Workspace isolation

Status: not yet run. Populated once `test_workspace_isolation`
(build order step 15) runs.

## Injection red team and out of scope scoring

Status: not yet run. Populated once the injection defense test
and the out of scope refusal scoring (build order steps 16 and 17) run.

## Limitations

This section is filled in at build order step 21, once every result
above is in hand, and names what was cut or substituted along the way,
including `evalset/public_domain/PROVENANCE.md`'s documented departure
from the original public domain document requirement.
