"""The package root exports exactly the names a composer needs."""

import theseus


def test_composition_entry_points_import_from_root():
    from theseus import (  # noqa: F401
        BufferPolicy,
        BufferedStimulusLog,
        CommandChannel,
        CommandFeed,
        HighWaterMarks,
        MemoryCommandChannel,
        ReentrantIngest,
        ReplicationIngress,
        SseCommandChannel,
    )

    for name in (
        "HighWaterMarks",
        "ReentrantIngest",
        "ReplicationIngress",
        "BufferedStimulusLog",
        "BufferPolicy",
        "CommandFeed",
        "CommandChannel",
        "MemoryCommandChannel",
        "SseCommandChannel",
        "command_type",
        "command_content",
        "is_command",
        "command_target",
    ):
        assert name in theseus.__all__
        assert hasattr(theseus, name)


def test_internals_stay_off_all():
    # __all__ is a compatibility commitment for tag-pinned consumers; it stays narrow.
    for name in ("parse_batch", "BatchRejected", "CoalescingTrigger", "plan_batch"):
        assert name not in theseus.__all__
