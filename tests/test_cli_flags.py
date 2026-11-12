import pathlib
import subprocess
import sys
import time

import pytest

SERVICE = str(pathlib.Path(__file__).resolve().parent.parent / "paddle_service.py")


def test_parser_rejects_bad_conf_gate(service):
    for bad in ("2", "0", "-0.1", "abc"):
        with pytest.raises(SystemExit):
            service.build_parser().parse_args(["--conf-gate", bad])
    assert service.build_parser().parse_args(["--conf-gate", "0.8"]).conf_gate == 0.8


def test_bad_flag_fails_before_heavy_imports():
    """Run as a script with NO paddle available: must fail on the flag, fast, not on ImportError."""
    t0 = time.time()
    r = subprocess.run([sys.executable, SERVICE, "--conf-gate", "2"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2
    assert "conf-gate" in r.stderr and "ModuleNotFoundError" not in r.stderr
    assert time.time() - t0 < 5
