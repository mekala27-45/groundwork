"""Chunking strategies over the same ExtractedDocument, compared rather
than assumed. See naive.py for the fixed window strategy and packing.py
for the token-budget logic every strategy shares.
"""

from groundwork_chunk.models import ChunkCandidate
from groundwork_chunk.naive import chunk_naive

__all__ = ["ChunkCandidate", "chunk_naive"]
