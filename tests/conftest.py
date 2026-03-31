"""Shared pytest fixtures for presidio-hardened-vol-assign tests."""

from pathlib import Path

import pytest


@pytest.fixture()
def fixtures_dir() -> Path:
    """Path to the tests/fixtures/ directory."""
    return Path(__file__).parent / "fixtures"
