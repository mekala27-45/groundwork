# groundwork

A multi tenant platform that turns an uploaded PDF into a chatbot that
answers questions grounded only in that document, with verifiable
citations, a measured refusal for anything the document does not cover,
and a full evaluation harness proving retrieval quality, chunking
strategy, and answer faithfulness rather than asserting them.

This build is still in progress. Every number in this section is
rendered by `scripts/check_published_numbers.py` from a live query
against this repository's own database and `evalset/questions.yaml`,
never typed in by hand; see `packages/api/src/groundwork_api/claims.py`
for exactly how, and RESULTS.md for the full evaluation writeup as it
fills in. ARCHITECTURE.md lands at build order step 26.

## Current state

- Demo workspaces seeded: 3
- Documents indexed: 6
- Chunks stored: 53 naive, 73 structure aware
- Golden evaluation set: 45 questions (12 direct, 8 boundary, 6 table, 6 cross_document, 8 out_of_scope, 5 injection)
- Retrieval evaluation harness: computed
- Faithfulness, isolation, and red team scorecard: computed

Full setup, deployment, and usage instructions land here once the API
and web app are built and deployed (build order steps 19 through 27).

## License

Apache 2.0. See LICENSE.
