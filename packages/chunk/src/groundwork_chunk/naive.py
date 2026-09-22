"""The naive strategy: fixed token windows over the whole document, no
awareness of section structure at all. This is what `chunk_size=1000` at
the library default looks like when someone builds it themselves instead
of importing it, which is the point of having it: section 7 of the day 5
prompt asks for a real comparison, not a straw man, so this has to be a
genuine, reasonably implemented baseline rather than something built to
lose.
"""

from __future__ import annotations

from groundwork_chunk.models import ChunkCandidate
from groundwork_chunk.packing import build_canonical_text, pack_atom_range, resolve_chunk_text
from groundwork_chunk.segments import build_atoms
from groundwork_chunk.tokenize import get_tokenizer_backend
from groundwork_ingest.models import ExtractedDocument

DEFAULT_CHUNK_SIZE_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 32


def chunk_naive(
    doc: ExtractedDocument,
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkCandidate]:
    atoms = build_atoms(doc)
    if not atoms:
        return []
    _, spans = build_canonical_text(atoms)
    ranges = pack_atom_range(
        atoms, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
    )
    backend = get_tokenizer_backend()

    chunks: list[ChunkCandidate] = []
    for start_idx, end_idx in ranges:
        text, char_start, char_end, page_start, page_end = resolve_chunk_text(
            atoms, spans, start_idx, end_idx
        )
        chunks.append(
            ChunkCandidate(
                text=text,
                page_start=page_start,
                page_end=page_end,
                char_start=char_start,
                char_end=char_end,
                section_title=None,
                strategy="naive",
                tokenizer_backend=backend,
            )
        )
    return chunks
