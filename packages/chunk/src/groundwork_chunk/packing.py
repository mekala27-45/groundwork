"""The token-budget packing algorithm shared by both strategies. Naive
chunking runs it once over the whole atom sequence; structure-aware
chunking runs it once per detected section. Sharing this function is what
makes "the only difference between the strategies is how they group atoms
first" literally true in the code, not just true in the write-up.

_separator is the one place that decides how two adjacent atoms join: a
table gets a blank line on either side so it reads as its own block, plain
words just get a space. build_canonical_text, pack_atom_range and
resolve_chunk_text all resolve text through it via _join_atoms, so the
packing decision and the text it produces can never drift apart. They used
to: pack_atom_range's budget check once joined candidates with a bare
" ".join(...) while resolve_chunk_text built the real text with the
blank-line rule, so a chunk sitting right at the budget with a table on its
boundary could pass the check and still come out over budget once resolved.
A hypothesis run on test_no_chunk_exceeds_the_token_budget_unless_it_is_a_
single_atom found it; sharing one join function is what makes it
structurally impossible to reintroduce.
"""

from __future__ import annotations

from groundwork_chunk.models import Atom
from groundwork_chunk.tokenize import count_tokens


def _separator(prev_atom: Atom, next_atom: Atom) -> str:
    return "\n\n" if prev_atom.kind == "table" or next_atom.kind == "table" else " "


def _join_atoms(atoms: list[Atom], start_idx: int, end_idx: int) -> str:
    """The exact text atoms[start_idx:end_idx] resolves to, separators
    included. The single source of truth both the packer's budget check
    and the caller-facing chunk text are built from."""
    parts: list[str] = []
    for i in range(start_idx, end_idx):
        if i > start_idx:
            parts.append(_separator(atoms[i - 1], atoms[i]))
        parts.append(atoms[i].text)
    return "".join(parts)


def build_canonical_text(atoms: list[Atom]) -> tuple[str, list[tuple[int, int]]]:
    """The text every chunk's char_start/char_end is an offset into, and
    the per-atom (start, end) span within it."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, atom in enumerate(atoms):
        if i > 0:
            sep = _separator(atoms[i - 1], atom)
            parts.append(sep)
            cursor += len(sep)
        start = cursor
        parts.append(atom.text)
        cursor += len(atom.text)
        spans.append((start, cursor))
    return "".join(parts), spans


def pack_atom_range(
    atoms: list[Atom],
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    """Group atoms[start:end] index ranges (end exclusive) so that no
    group's resolved text exceeds chunk_size_tokens, except a single atom
    that alone exceeds it (an atom, in particular a table, is never split,
    even if that means one chunk runs over budget), and consecutive groups
    share up to overlap_tokens of trailing word atoms. Returns index
    ranges only; the caller resolves them to text and character offsets so
    this function stays agnostic to how the canonical text was built.
    """
    n = len(atoms)
    if n == 0:
        return []

    ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        # The first atom is always included regardless of its own size
        # (j > i is False on the loop's first pass, so the budget check
        # below is skipped for it), which is what guarantees forward
        # progress: j is always at least i + 1 by the time this inner loop
        # exits, even when that first atom alone is already over budget.
        j = i
        while j < n:
            if j > i and count_tokens(_join_atoms(atoms, i, j + 1)) > chunk_size_tokens:
                break
            j += 1

        ranges.append((i, j))
        if j >= n:
            break

        # Seed the next chunk with up to overlap_tokens of trailing word
        # atoms from the chunk just closed. Never seed with a table: a
        # table either belongs entirely to one chunk or, through overlap,
        # would be duplicated whole into the next, which is not what
        # "overlap" is supposed to mean for a unit that large.
        overlap_start = j
        acc = 0
        k = j - 1
        while k >= i and acc < overlap_tokens and atoms[k].kind == "word":
            acc += count_tokens(atoms[k].text)
            overlap_start = k
            k -= 1

        i = overlap_start if overlap_start > i else j

    return ranges


def resolve_chunk_text(
    atoms: list[Atom],
    spans: list[tuple[int, int]],
    start_idx: int,
    end_idx: int,
) -> tuple[str, int, int, int, int]:
    """Returns (text, char_start, char_end, page_start, page_end) for the
    atom slice [start_idx, end_idx)."""
    char_start = spans[start_idx][0]
    char_end = spans[end_idx - 1][1]
    page_start = min(a.page_number for a in atoms[start_idx:end_idx])
    page_end = max(a.page_number for a in atoms[start_idx:end_idx])
    text = _join_atoms(atoms, start_idx, end_idx)
    return text, char_start, char_end, page_start, page_end
