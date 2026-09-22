"""Shared primitives used by every groundwork package.

StrictModel, the UUIDv7 id helper, and the settings base class all started
in trajectory (day 1) and have now been ported forward unchanged through
pricepoint, cityflow and frontdesk. Nothing here is new to this build.
"""

from groundwork_core.ids import new_id
from groundwork_core.model import StrictModel
from groundwork_core.redaction import redact_for_log

__all__ = ["StrictModel", "new_id", "redact_for_log"]
