"""Two chunking strategies over the same ExtractedDocument, compared
rather than assumed. See naive.py and structure.py for the strategies
themselves and packing.py for the token-budget logic both share.
"""

from groundwork_chunk.models import ChunkCandidate
from groundwork_chunk.naive import chunk_naive
from groundwork_chunk.structure import chunk_structure

__all__ = ["ChunkCandidate", "chunk_naive", "chunk_structure"]
