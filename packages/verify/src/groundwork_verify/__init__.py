"""Independent verification: faithfulness checking by a locally run NLI
model that never calls back into the model or system that produced the
answer being checked. See split.py for turning an answer into sentence
level claims, and faithfulness.py for scoring each one against its cited
chunk.
"""

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
from groundwork_verify.split import Claim, split_claims

__all__ = [
    "ENTAILMENT_OVERLAP_THRESHOLD",
    "Claim",
    "ClaimVerification",
    "CrossEncoderFaithfulnessScorer",
    "FaithfulnessBackend",
    "FaithfulnessScorer",
    "LexicalFaithfulnessScorer",
    "check_faithfulness",
    "get_faithfulness_scorer",
    "reset_faithfulness_scorer_cache",
    "resolve_faithfulness_backend",
    "split_claims",
]
