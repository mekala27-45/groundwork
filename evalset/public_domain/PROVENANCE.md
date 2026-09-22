# Provenance: the second demo workspace's document

## What the day 5 spec asked for

Part 4 of the build prompt calls for one U.S. federal government
publication for the second demo workspace: categorically public domain,
at least fifteen pages, with at least one multi page table and a multi
column section. A Small Business Administration guide or a NIST
technical publication were both named as good fits.

## What actually happened

Three candidate documents were identified by domain (an SBA small
business guide, a NIST publication, an IRS publication), each a genuine
match for the spec. All three failed to download. This build environment's
outbound network access is proxied through an allowlist covering package
registries and a small set of specific hosts; `irs.gov`, `nist.gov`, and
`sba.gov` are not on it. Direct requests to all three were attempted and
every one was rejected by the proxy at the TLS tunnel stage, the same
failure shape already hit once earlier in this build when
`huggingface.co`'s blob storage host was unreachable for a cross encoder
download. `WebSearch` and `WebFetch` could describe and summarize these
documents but neither tool can deliver raw PDF bytes into this sandbox,
and routing around a blocked fetch with a direct download is against this
project's own operating rules.

Two options remained. Reproduce one of these real documents from memory
was rejected: an eval workspace's entire value is that its facts are
known and checkable, and a publication retyped from memory risks
confidently wrong figures presented with a real agency's name attached to
them, which is worse than no document at all. Silently substituting
different content while still citing a federal agency was rejected for
the same reason in the other direction: it would be a false provenance
claim sitting in a project whose whole thesis is that RAG output should
be checkable against its real source.

What this workspace uses instead is original content, written for this
project and stated as such here and in `README.md`: **"A Field Reference
Guide to Evaluating Machine Learning Systems,"** a technical reference
covering classification, regression, and ranking metrics; statistical
significance testing; LLM and agentic evaluation; fairness and subgroup
analysis; and a 48 row metric reference table plus a 37 entry two column
glossary. It is not a U.S. government publication, not attributed to any
federal agency, and not claimed to be public domain by virtue of
government authorship. It is original material by this project's author,
and is offered under the same Apache 2.0 license as the rest of the
repository.

## Why this still proves what section 1 of the spec needs proven

The spec's second document exists to demonstrate that the ingestion and
chunking pipeline generalizes past Meridian Coaching's own layout, not to
demonstrate access to a specific government archive. The substitute
document is deliberately built to differ from every Meridian document
along the same axes a real second source would: a different domain
(machine learning evaluation rather than life coaching), a longer page
count (17 pages against Meridian's 1 to 2 pages per document), a genuine
multi page table (a metric reference table spanning pages 11 through 13),
and a genuine two column glossary section, a layout no Meridian document
uses at all. `packages/ingest/tests` and the eval set's table lookup
category exercise this document specifically because of that structural
difference, not in spite of the substitution.

## What this means for anyone extending this project

If this project moves to an environment with unrestricted network
access, swapping in a real federal publication is a content change, not
an architecture change: replace `scripts/build_reference_guide.py`'s
role with a downloaded PDF, regenerate the chunk and embedding rows for
that workspace, and update `evalset/questions.yaml`'s table lookup and
boundary spanning questions to match the new document's real content.
Nothing in `packages/ingest`, `packages/chunk`, or `packages/retrieve`
assumes anything about where the second workspace's document came from.

## Generation

`evalset/public_domain/ml-eval-reference-guide.pdf` is built by
`scripts/build_reference_guide.py`, itself built on the shared PDF
layout helper in `scripts/pdf_builder.py`. Regenerate it with:

```
uv run python scripts/build_reference_guide.py
```
