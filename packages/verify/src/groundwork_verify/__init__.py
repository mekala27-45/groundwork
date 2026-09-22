"""Independent verification: faithfulness checking by a locally run NLI
model that never calls back into the model or system that produced the
answer being checked, deterministic citation span verification against
the database, deterministic out of scope refusal scoring, and secondary
quality scoring kept clearly apart from all three. See split.py for
turning an answer into sentence level claims, faithfulness.py for scoring
each one against its cited chunk, citation.py for confirming a citation's
chunk is real, in-workspace, and unchanged since generation time,
refusal.py for checking whether an out of scope question was actually
refused rather than answered from outside knowledge, and quality.py for
the LiteLLMJudge that scores reply clarity and helpfulness only, never
faithfulness, and never the pass or fail gate.
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
from groundwork_verify.quality import (
    Judge,
    LiteLLMJudge,
    QualityBackend,
    QualityScore,
    parse_quality_score,
    resolve_quality_backend,
    score_reply_quality,
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
    "Judge",
    "LexicalFaithfulnessScorer",
    "LiteLLMJudge",
    "QualityBackend",
    "QualityScore",
    "check_faithfulness",
    "get_faithfulness_scorer",
    "is_out_of_scope_refusal",
    "parse_quality_score",
    "reset_faithfulness_scorer_cache",
    "resolve_faithfulness_backend",
    "resolve_quality_backend",
    "score_reply_quality",
    "split_claims",
    "verify_citations",
]
