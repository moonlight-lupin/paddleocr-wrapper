"""Test harness: stub paddle/paddleocr so paddle_service imports without a GPU stack."""
import copy
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_paddle = types.ModuleType("paddle")
_paddle.__version__ = "0.0-test"
_paddle.device = types.SimpleNamespace(
    is_compiled_with_cuda=lambda: False,
    cuda=types.SimpleNamespace(device_count=lambda: 0),
)
_paddleocr = types.ModuleType("paddleocr")
_paddleocr.__version__ = "0.0-test"


class _StubPaddleOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, img):
        return []


_paddleocr.PaddleOCR = _StubPaddleOCR
sys.modules.setdefault("paddle", _paddle)
sys.modules.setdefault("paddleocr", _paddleocr)

import paddle_service as svc  # noqa: E402

_PRISTINE = copy.deepcopy({k: v for k, v in svc.state.items()})


@pytest.fixture(autouse=True)
def fresh_state():
    """Reset the module-level state dict around every test."""
    svc.state.clear()
    svc.state.update(copy.deepcopy(_PRISTINE))
    svc.LOGS.clear()
    yield svc.state
    svc.state.clear()
    svc.state.update(copy.deepcopy(_PRISTINE))


@pytest.fixture
def service():
    return svc


class FakeModel:
    """Stands in for a PaddleOCR instance; returns one canned paddleocr-3.x style result."""

    def __init__(self, texts=("hello",), scores=(0.99,), fail=None):
        self.texts, self.scores, self.fail = list(texts), list(scores), fail
        self.calls = 0

    def predict(self, img):
        self.calls += 1
        if self.fail:
            raise self.fail
        polys = [[[0, i * 10], [9, i * 10], [9, i * 10 + 5], [0, i * 10 + 5]] for i in range(len(self.texts))]
        return [{"rec_texts": self.texts, "rec_scores": self.scores, "rec_polys": polys}]
