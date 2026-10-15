"""Headless import test for ``pygformula``."""

import os
import subprocess
import sys


def test_import_pygformula_without_display():
    """``import pygformula`` succeeds without DISPLAY, WAYLAND_DISPLAY or MPLBACKEND.

    ``plot.py`` used to force the Qt5Agg backend unconditionally at import time,
    which failed on machines without a display or without Qt. This test imports
    the package in a fresh subprocess with the display and backend environment
    variables removed, so the pytest-level ``Agg`` workaround in ``conftest.py``
    does not apply.
    """
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("MPLBACKEND", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import pygformula"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
