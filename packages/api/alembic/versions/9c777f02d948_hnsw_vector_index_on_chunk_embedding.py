"""hnsw vector index on chunk embedding

pgvector cannot use a plain btree index for a nearest neighbour ORDER BY,
so test_pgvector_cosine_distance_orders_by_similarity (packages/api/tests/
test_models.py) passed against the initial migration only because its
fixture data is three rows: a full sequential scan finds the right order
regardless of an index. HNSW over vector_cosine_ops is the same operator
class that query already uses (Chunk.embedding.cosine_distance), chosen
over ivfflat because it needs no training step and stays effective on a
table this size without a minimum row count to be useful, both of which
matter for a demo dataset rather than a warehouse-scale one.

Revision ID: 9c777f02d948
Revises: 328787185122
Create Date: 2026-09-21 23:10:22.507280
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "9c777f02d948"
down_revision: str | None = "328787185122"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

INDEX_NAME = "ix_chunk_embedding_hnsw_cosine"


def upgrade() -> None:
    op.create_index(
        INDEX_NAME,
        "chunk",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="chunk")
