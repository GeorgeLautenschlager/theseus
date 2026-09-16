"""Durable metadata and recall rendering for extracted claims."""

from __future__ import annotations


ATTRIBUTIONS = frozenset({"direct_observation", "partner_report", "inference"})
ACTION_STATUSES = frozenset({"not_applicable", "intention", "attempt", "failure", "confirmed_outcome"})


def render_metadata(
    support_event_ids: tuple[str, ...] | None,
    attribution: str | None,
    reported_by: str | None,
    action_status: str | None,
) -> str:
    provenance = ", ".join(support_event_ids) if support_event_ids else "unknown"
    parts = [f"Supporting events: {provenance}"]
    if attribution is not None:
        source = f" ({reported_by})" if reported_by else ""
        parts.append(f"Attribution: {attribution}{source}")
    if action_status is not None and action_status != "not_applicable":
        parts.append(f"Action status: {action_status}")
    return " [" + "; ".join(parts) + "]"
