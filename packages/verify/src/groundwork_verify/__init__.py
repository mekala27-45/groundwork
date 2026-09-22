"""Independent verification: faithfulness checking by a locally run NLI
model that never calls back into the model or system that produced the
answer being checked, deterministic citation span verification against
the database, and deterministic out of scope refusal scoring. See
split.py for turning an answer into sentence level claims, faithfulness.py
for scoring each one against its cited chunk, citation.py for confirming
a citation's chunk is real, in-workspace, and unchanged since generation
time, and refusal.py for checking whether an out of scope question was
actually refused rather than answered from outside knowledge.
"""

from groundwork_verify.citation import CitationVerification, verify_citations
from groundwork_verify.faithfulness import (
    ENTAILMENT_OVERLAP_THRESHOLD,
    ClaimVerification,
    CrossEncoderFaithfulnessScorer,
    FaithfulnessBackend,
    FaithfulnessScorer,
    LexicalFaithfulnessScorer,
    check_faithfulness,
    get_faithfulness_scorer,
    reset_faithfulness_scorer_cache,
    resolve_faithfulness_backend,
)
from groundwork_verify.refusal import is_out_of_scope_refusal
from groundwork_verify.split import Claim, split_claims

__all__ = [
    "ENTAILMENT_OVERLAP_THRESHOLD",
    "Claim",
    "ClaimVerification",
    "CitationVerification",
    "CrossEncoderFaithfulnessScorer",
    "FaithfulnessBackend",
    "FaithfulnessScorer",
    "LexicalFaithfulnessScorer",
    "check_faithfulness",
    "get_faithfulness_scorer",
    "is_out_of_scope_refusal",
    "reset_faithfulness_scorer_cache",
    "resolve_faithfulness_backend",
    "split_claims",
    "verify_citations",
]
