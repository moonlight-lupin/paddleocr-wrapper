import numpy as np
import pytest
from fastapi import HTTPException

from conftest import FakeModel

IMG = np.zeros((10, 10, 3), dtype=np.uint8)


def fields(conf):
    return {"lines": [{"index": 0, "text": "x", "confidence": conf, "box": None}], "text": "x"}


# ---- result parsing -------------------------------------------------------

def test_result_fields_basic(service):
    out = service.results_to_fields(FakeModel(["a", "b"], [0.9, 0.8]).predict(IMG))
    assert out["text"] == "a\nb"
    assert [ln["confidence"] for ln in out["lines"]] == [0.9, 0.8]
    assert out["lines"][0]["box"] == [[0.0, 0.0], [9.0, 0.0], [9.0, 5.0], [0.0, 5.0]]


def test_empty_results_do_not_raise(service):
    assert service.results_to_fields([]) == {"lines": [], "text": ""}


# ---- confidence gate ------------------------------------------------------

class StubWorker:
    def __init__(self, f):
        self.f = f
        self.proc = None

    def call(self, img):
        return (self.f, "rapidocr-cpu") if self.f is not None else (None, "nope")


@pytest.mark.parametrize("conf,passes", [(0.99, True), (0.70, True), (0.6999, False), (None, False)])
def test_gate(service, fresh_state, conf, passes):
    fresh_state["rapid_worker"] = StubWorker(fields(conf))
    out, why = service.rapid_fields(IMG)
    assert (out is not None) is passes
    assert fresh_state["rapid"]["gated"] == (0 if passes else 1)


def test_gate_zero_lines_is_gated(service, fresh_state):
    fresh_state["rapid_worker"] = StubWorker({"lines": [], "text": ""})
    assert service.rapid_fields(IMG)[0] is None


def test_circuit_opens_after_two_transport_failures(service, fresh_state):
    fresh_state["rapid_worker"] = StubWorker(None)
    service.rapid_fields(IMG)
    assert not fresh_state["rapid"]["circuit_open"]
    service.rapid_fields(IMG)
    assert fresh_state["rapid"]["circuit_open"]
    assert service.rapid_fields(IMG)[1] == "tier2 circuit open"


# ---- chain routing --------------------------------------------------------

def test_gpu_failure_descends_to_tier2_in_auto(service, fresh_state):
    fresh_state.update(model=FakeModel(fail=RuntimeError("oom")), device="gpu:0", requested_device="auto",
                       rapid_worker=StubWorker(fields(0.99)))
    out, label = service.ensure_fields(IMG)
    assert label == "rapidocr-cpu"
    assert fresh_state["fallback_active"] and "oom" in fresh_state["fallback_reason"]


def test_gpu_failure_with_explicit_gpu_is_500(service, fresh_state):
    fresh_state.update(model=FakeModel(fail=RuntimeError("oom")), device="gpu:0", requested_device="gpu",
                       rapid_worker=StubWorker(fields(0.99)))
    with pytest.raises(HTTPException) as e:
        service.ensure_fields(IMG)
    assert e.value.status_code == 500


def test_gpu_recovery_clears_fallback(service, fresh_state):
    fresh_state.update(model=FakeModel(), device="gpu:0", requested_device="auto",
                       fallback_active=True, fallback_reason="old")
    _, label = service.ensure_fields(IMG)
    assert label == "gpu:0" and not fresh_state["fallback_active"] and fresh_state["fallback_reason"] is None


def test_gated_tier2_reruns_on_paddle_cpu(service, fresh_state, monkeypatch):
    cpu = FakeModel(["right"], [0.99])
    fresh_state.update(model=FakeModel(fail=RuntimeError("x")), device="gpu:0", requested_device="auto",
                       rapid_worker=StubWorker(fields(0.5)))
    monkeypatch.setattr(service, "init_model", lambda dev: cpu)
    monkeypatch.setattr(service, "warm_predict", lambda m: True)
    out, label = service.ensure_fields(IMG)
    assert label == "cpu" and out["text"] == "right"
    assert fresh_state["model"] is not cpu          # tier 3 never displaces the boot model
    assert fresh_state["cpu_model"] is cpu


def test_cpu_booted_auto_prefers_tier2(service, fresh_state):
    cpu = FakeModel()
    fresh_state.update(model=cpu, device="cpu", requested_device="auto", rapid_worker=StubWorker(fields(0.99)))
    assert service.ensure_fields(IMG)[1] == "rapidocr-cpu" and cpu.calls == 0


def test_explicit_cpu_never_uses_tier2(service, fresh_state):
    fresh_state.update(model=FakeModel(), device="cpu", requested_device="cpu", rapid_worker=StubWorker(fields(0.99)))
    assert service.ensure_fields(IMG)[1] == "cpu"


def test_engine_chain_labels(service, fresh_state):
    assert service.engine_chain() == ["paddle-gpu", "paddle-cpu"]
    fresh_state["rapid_worker"] = StubWorker(None)
    assert service.engine_chain() == ["paddle-gpu", "rapidocr-cpu", "paddle-cpu"]
