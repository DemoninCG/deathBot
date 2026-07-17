"""Tests for the dedicated deterministic evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from evaluate import resolve_weights  # noqa: E402


def test_missing_weights_fail_instead_of_using_random_network():
    with pytest.raises(FileNotFoundError):
        resolve_weights("definitely_missing_weights.pt")