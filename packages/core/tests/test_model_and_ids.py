from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from groundwork_core import StrictModel, new_id
from groundwork_core.redaction import content_hash, redact_for_log


class Sample(StrictModel):
    name: str
    count: int


def test_strict_model_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Sample(name="a", count=1, extra_field="not allowed")  # type: ignore[call-arg]


def test_strict_model_validates_on_assignment() -> None:
    s = Sample(name="a", count=1)
    with pytest.raises(ValidationError):
        s.count = "not an int"  # type: ignore[assignment]


def test_new_id_is_uuid7_and_monotonic() -> None:
    ids = [new_id() for _ in range(200)]
    assert all(isinstance(i, uuid.UUID) for i in ids)
    assert all(i.version == 7 for i in ids)
    assert ids == sorted(ids), "UUIDv7 ids must sort in creation order"
    assert len(set(ids)) == len(ids), "ids must be unique"


def test_redact_for_log_replaces_only_named_text_fields() -> None:
    payload = {"chunk_id": "abc", "text": "the quick brown fox", "page": 3}
    redacted = redact_for_log(payload, text_fields=("text",))
    assert redacted["chunk_id"] == "abc"
    assert redacted["page"] == 3
    assert redacted["text"] == {"length": 19, "sha256_16": content_hash("the quick brown fox")}
    assert "quick brown fox" not in str(redacted)


def test_redact_for_log_is_deterministic() -> None:
    a = redact_for_log({"text": "same text"}, text_fields=("text",))
    b = redact_for_log({"text": "same text"}, text_fields=("text",))
    assert a == b
