import sys
import pathlib

import numpy as np
import pytest

CHILD = str(pathlib.Path(__file__).parent / "fake_rapid_child.py")
IMG = np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def make_worker(service, monkeypatch):
    made = []

    def _make(mode="ok", conf="0.99", call_timeout=10, spawn_timeout=10):
        monkeypatch.setenv("FAKE_MODE", mode)
        monkeypatch.setenv("FAKE_CONF", conf)
        w = service.RapidWorker(sys.executable, CHILD)
        w.CALL_TIMEOUT_S = call_timeout
        w.SPAWN_TIMEOUT_S = spawn_timeout
        made.append(w)
        return w

    yield _make
    for w in made:
        w.stop()


def test_roundtrip(make_worker):
    w = make_worker()
    fields, label = w.call(IMG)
    assert label == "rapidocr-cpu" and fields["text"] == "hi"
    assert w.call(IMG)[0] is not None            # child is reused


def test_noise_before_ready_is_skipped(make_worker):
    assert make_worker("noise").probe() is None


def test_probe_reports_child_dying_before_ready(make_worker):
    assert "died before READY" in make_worker("die_before_ready").probe()


def test_per_image_failure_keeps_child_alive(make_worker):
    w = make_worker("ocr_error")
    fields, why = w.call(IMG)
    assert fields is None and "OCR failed" in why
    assert w.proc is not None and w.proc.poll() is None


def test_hung_child_is_bounded_and_killed(make_worker):
    import time
    w = make_worker("hang", call_timeout=1.5)
    t0 = time.time()
    fields, why = w.call(IMG)
    assert fields is None and "TIMED-OUT" in why
    assert time.time() - t0 < 5
    assert w.proc is None


def test_dying_child_gives_up_after_one_respawn(make_worker, service):
    w = make_worker("die_on_call")
    fields, why = w.call(IMG)
    assert fields is None
    assert service.state["rapid"]["failures"] == 2   # initial attempt + one respawn
