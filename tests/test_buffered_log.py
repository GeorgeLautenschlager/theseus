"""BufferPolicy's contract: size and hysteresis, validated at composition time."""

from __future__ import annotations

import dataclasses

import pytest

from theseus.surrogates.buffer import BufferPolicy


def test_defaults_are_the_documented_ones():
    policy = BufferPolicy()
    assert policy.max_bytes == 256 * 1024 * 1024
    assert policy.low_water == 0.8


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_max_bytes_refused(value):
    with pytest.raises(ValueError, match="max_bytes"):
        BufferPolicy(max_bytes=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 1.5])
def test_low_water_outside_open_interval_refused(value):
    with pytest.raises(ValueError, match="low_water"):
        BufferPolicy(low_water=value)


@pytest.mark.parametrize("value", [0.99, 0.01])
def test_legal_boundary_values_accepted(value):
    assert BufferPolicy(low_water=value).low_water == value


def test_policy_is_frozen():
    policy = BufferPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_bytes = 1  # type: ignore[misc]
