"""Durable metadata and recall rendering for extracted claims."""

from __future__ import annotations


ATTRIBUTIONS = frozenset({"direct_observation", "partner_report", "inference"})
ACTION_STATUSES = frozenset({"not_applicable", "intention", "attempt", "failure", "confirmed_outcome"})

# How a newly extracted fact relates to knowledge already on file, decided by
# the reconciliation step (see memory_module._reconcile_facts):
#   new           - nothing existing describes the same attribute.
#   reinforce     - restates an existing current record's value; not a change.
#   replace       - a genuine update to the same attribute (any wording).
#   coexist       - a different attribute under a shared/broad predicate; both
#                   stay current.
#   contradiction - conflicts with a current record and neither is clearly
#                   authoritative; both stay current, visibly in tension.
#   historical    - describes a past/superseded state relative to what's
#                   already current; kept in history, never becomes current.
RECONCILIATION_DECISIONS = frozenset({
    "new", "reinforce", "replace", "coexist", "contradiction", "historical",
})

# Promotion policy for inferred principles (see memory_module._reconcile_principles):
# an explicitly attributed preference (attribution != "inference" — the agent was
# told, or directly observed, not guessing) is established the moment it is
# written, no repetition required. A principle the agent itself inferred stays
# "provisional" — visibly unconfirmed — until independent evidence from at least
# this many distinct episodes has reinforced it. Retries of the same episode,
# repeated recall, and duplicate extraction within one episode never advance this:
# only a new episode id added to `supporting_episode_ids` counts.
WISDOM_STATUSES = frozenset({"provisional", "established"})
WISDOM_PROMOTION_THRESHOLD = 2


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
