"""Grounded generation with citations, and the zero cost extractive
fallback that keeps the system fully answerable with no LLM key
configured. See ledger.py for the spend ceiling every real call checks
before it runs, and generate.py for the two Generator implementations and
generate_answer(), the entry point that resolves between them.
"""

from groundwork_generate.generate import (
    GENERATION_SYSTEM_PROMPT,
    NOT_COVERED_MESSAGE,
    ExtractiveGenerator,
    GenerationBackend,
    GenerationResult,
    Generator,
    LiteLLMGenerator,
    generate_answer,
    resolve_generation_backend,
)
from groundwork_generate.ledger import SpendLedger, get_ledger, reset_ledger_cache

__all__ = [
    "GENERATION_SYSTEM_PROMPT",
    "NOT_COVERED_MESSAGE",
    "ExtractiveGenerator",
    "GenerationBackend",
    "GenerationResult",
    "Generator",
    "LiteLLMGenerator",
    "SpendLedger",
    "generate_answer",
    "get_ledger",
    "resolve_generation_backend",
    "reset_ledger_cache",
]
