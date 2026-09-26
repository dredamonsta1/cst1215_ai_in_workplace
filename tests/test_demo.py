"""Smoke test for the presentation script.

demo.py is only ever exercised by a human in front of an audience, which is
the worst possible time to discover a typo. This runs the whole thing
headlessly and asserts the narrative beats all appear.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_demo(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "demo.py", "--no-color", *extra],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_demo_runs_end_to_end():
    result = _run_demo()
    assert result.returncode == 0, result.stderr


def test_demo_reaches_every_act():
    result = _run_demo()
    for act in range(1, 8):
        assert f"ACT {act}." in result.stdout, f"act {act} missing"


def test_demo_shows_the_headline_routing_shift():
    """The whole point of the talk: the recommendation moves when latency tightens."""
    stdout = _run_demo().stdout
    assert "eu-north-se" in stdout
    assert "The recommendation shifted to us-central-ia" in stdout
    assert "93.5% reduction" in stdout


def test_demo_hides_server_stderr_by_default():
    """The deliberate error in the last act must not print above act 1."""
    result = _run_demo()
    assert "Tool 'get_optimal_region' failed" not in result.stderr
    assert "Tool 'get_optimal_region' failed" not in result.stdout
    # But the error is still narrated in-band, via the protocol response.
    assert "no region satisfies max_latency_ms=5" in result.stdout


def test_demo_can_surface_server_log_on_request():
    result = _run_demo("--server-log")
    assert result.returncode == 0
    assert "no region satisfies max_latency_ms=5" in result.stderr
