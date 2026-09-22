"""The structure-aware strategy: split on detected section headers first,
then further split any section that still exceeds the token budget, with
overlap, using the identical packing function naive.py uses. Never splits
a table, exactly like naive, because that guarantee lives in packing.py
rather than in either strategy.
"""

from __future__ import annotations

from groundwork_chunk.models import Atom, ChunkCandidate
from groundwork_chunk.packing import build_canonical_text, pack_atom_range, resolve_chunk_text
from groundwork_chunk.segments import build_atoms
from groundwork_chunk.tokenize import get_tokenizer_backend
from groundwork_ingest.models import ExtractedDocument

DEFAULT_CHUNK_SIZE_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 32
MAX_TITLE_WORDS = 12


def _find_sections(atoms: list[Atom]) -> list[tuple[str | None, int, int]]:
    """Returns (title, start_idx, end_idx) for each section, end exclusive.
    A leading section with no header gets title=None."""
    header_run_starts: list[int] = []
    in_header = False
    for i, atom in enumerate(atoms):
        if atom.is_section_start and not in_header:
            header_run_starts.append(i)
            in_header = True
        elif not atom.is_section_start:
            in_header = False

    if not header_run_starts:
        return [(None, 0, len(atoms))] if atoms else []

    sections: list[tuple[str | None, int, int]] = []
    if header_run_starts[0] > 0:
        sections.append((None, 0, header_run_starts[0]))

    for idx, start in enumerate(header_run_starts):
        end = header_run_starts[idx + 1] if idx + 1 < len(header_run_starts) else len(atoms)
        title_words: list[str] = []
        j = start
        while j < end and atoms[j].is_section_start and len(title_words) < MAX_TITLE_WORDS:
            title_words.append(atoms[j].text)
            j += 1
        title = " ".join(title_words) if title_words else None
        sections.append((title, start, end))

    return sections


def chunk_structure(
    doc: ExtractedDocument,
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkCandidate]:
    atoms = build_atoms(doc)
    if not atoms:
        return []
    _, spans = build_canonical_text(atoms)
    sections = _find_sections(atoms)
    backend = get_tokenizer_backend()

    chunks: list[ChunkCandidate] = []
    for title, sec_start, sec_end in sections:
        section_atoms = atoms[sec_start:sec_end]
        local_ranges = pack_atom_range(
            section_atoms, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
        )
        for local_start, local_end in local_ranges:
            start_idx = sec_start + local_start
            end_idx = sec_start + local_end
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
                    section_title=title,
                    strategy="structure",
                    tokenizer_backend=backend,
                )
            )
    return chunks
