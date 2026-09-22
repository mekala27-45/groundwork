"""StrictModel: the one base class every schema in this repository uses.

Ported from trajectory (day 1) unchanged. Extra fields are forbidden so a
typo in a request body fails loudly instead of being silently dropped,
assignment is validated so mutating a model after construction cannot put
it in an invalid state, and computed fields round-trip through
model_dump / model_validate rather than being recomputed accidentally.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base class for every request, response and stored-record schema."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=False,
    )
