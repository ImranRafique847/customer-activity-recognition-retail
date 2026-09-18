"""
Tests for code/inference.py (SageMaker inference handler).

Covers tasks 9.1–9.11:
  - model_fn: loads model, validates class count, sets _ERROR_STATE on mismatch
  - input_fn: decodes JPEG/PNG, returns float32 tensor [1,3,416,416] in [0,1]
  - predict_fn: runs model, returns dict with inference_time_ms
  - output_fn: serialises to JSON, handles _ERROR_STATE, handles predict_fn exceptions

Requirements: 5.6, 5.7, 5.8, 5.9, 5.11, 8.2, 9.4
"""

import importlib.util
import io
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Absolute path to inference.py — avoids collision with stdlib 'code' module
_INFERENCE_PY = Path(__file__).parent.parent / "code" / "inference.py"


# ---------------------------------------------------------------------------
# Module-level import helpers
# ---------------------------------------------------------------------------
# inference.py imports torch, ultralytics, PIL, numpy at module level.
# We mock these at sys.modules before importing to avoid needing GPU packages
# in CI.  Each call to _load_inference_module() produces a fresh module object
# with a clean _ERROR_STATE.

def _make_torch_stub():
    """Minimal torch stub sufficient for inference.py."""
    torch = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data, shape=None):
            self._data = data
            self._shape = shape or [1, 3, 416, 416]

        @property
        def shape(self):
            return self._shape

        def tolist(self):
            return self._data if isinstance(self._data, list) else [self._data]

        def permute(self, *dims):
            return self

        def unsqueeze(self, dim):
            return self

    torch.Tensor = _Tensor

    def from_numpy(arr):
        return _Tensor(arr)

    torch.from_numpy = from_numpy
    return torch


def _make_numpy_stub():
    """Minimal numpy stub — only array() and float32 needed."""
    np = types.ModuleType("numpy")

    class _NdArray:
        def __init__(self, data):
            self._data = data
            self.shape = (416, 416, 3)

        def __truediv__(self, other):
            return self  # normalisation: return self unchanged

        def permute(self, *args):
            return self

        def unsqueeze(self, dim):
            return self

    class _Array(_NdArray):
        pass

    np.float32 = float

    def array(img, dtype=None):
        return _NdArray(img)

    np.array = array
    return np


def _make_pil_stub():
    """Minimal PIL/Pillow stub."""
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")

    class _FakeImage:
        def convert(self, mode):
            return self

        def resize(self, size):
            return self

    def open(fp):
        return _FakeImage()

    pil_image.open = open
    pil.Image = pil_image
    return pil, pil_image


def _make_ultralytics_stub(nc=5):
    """Minimal ultralytics stub with a configurable class count."""
    ul = types.ModuleType("ultralytics")

    class _Model:
        def __init__(self, path):
            self.model = MagicMock()
            self.model.nc = nc
            self.names = {i: f"class_{i}" for i in range(nc)}

        def __call__(self, tensor):
            return [_FakeResult(nc)]

    class _FakeBox:
        def __init__(self):
            self.xyxyn = [[0.1, 0.2, 0.3, 0.4]]
            self.cls = [MagicMock(item=lambda: 0)]
            self.conf = [MagicMock(item=lambda: 0.9)]

        def tolist(self):
            return [0.1, 0.2, 0.3, 0.4]

    class _FakeResult:
        def __init__(self, nc_):
            self.names = {i: f"class_{i}" for i in range(nc_)}
            box = _FakeBox()
            # Make cls and conf behave like tensors with .item()
            box.cls = [MagicMock()]
            box.cls[0].item = MagicMock(return_value=0)
            box.conf = [MagicMock()]
            box.conf[0].item = MagicMock(return_value=0.9)
            box.xyxyn = [MagicMock()]
            box.xyxyn[0].tolist = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
            self.boxes = box

    ul.YOLO = _Model
    return ul


def _load_inference_module(nc_model=5, nc_config=5):
    """
    Load code/inference.py with all heavy dependencies mocked, returning a
    fresh module object each time (independent _ERROR_STATE).

    Uses importlib.util.spec_from_file_location to avoid the stdlib 'code'
    module name collision.

    nc_model  — number of classes the stubbed YOLO model reports
    nc_config — num_classes value in the mocked model_config.json
    """
    torch_stub = _make_torch_stub()
    np_stub = _make_numpy_stub()
    pil_stub, pil_image_stub = _make_pil_stub()
    ul_stub = _make_ultralytics_stub(nc=nc_model)

    config_content = json.dumps({
        "num_classes": nc_config,
        "class_names": [f"class_{i}" for i in range(nc_config)],
        "input_size": 416,
    })

    mocked_modules = {
        "torch": torch_stub,
        "numpy": np_stub,
        "PIL": pil_stub,
        "PIL.Image": pil_image_stub,
        "ultralytics": ul_stub,
    }

    # Give each load a unique module name so Python doesn't cache the old one
    import time as _time
    unique_name = f"_inference_fresh_{id(mocked_modules)}_{_time.monotonic_ns()}"

    spec = importlib.util.spec_from_file_location(unique_name, _INFERENCE_PY)
    inf = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, mocked_modules):
        with patch("pathlib.Path.read_text", return_value=config_content):
            spec.loader.exec_module(inf)

    # Ensure clean slate
    inf._ERROR_STATE = False
    inf._MODEL_RUN_NAME = "unknown"
    return inf


# ---------------------------------------------------------------------------
# model_fn
# ---------------------------------------------------------------------------


class TestModelFn:
    """Tests for model_fn (Req 5.6, 9.4)."""

    def test_returns_model_object(self, tmp_path):
        """model_fn returns a non-None callable model."""
        inf = _load_inference_module(nc_model=5, nc_config=5)
        model_dir = str(tmp_path)
        # Create a fake best.pt so YOLO(path) doesn't complain
        (tmp_path / "best.pt").write_bytes(b"fake")
        (tmp_path / "code").mkdir()
        (tmp_path / "code" / "model_config.json").write_text(
            json.dumps({"num_classes": 5, "class_names": ["a"] * 5, "input_size": 416})
        )
        with patch.dict(sys.modules, {
            "ultralytics": _make_ultralytics_stub(nc=5),
        }):
            model = inf.model_fn(model_dir)
        assert model is not None

    def test_error_state_false_on_class_count_match(self, tmp_path):
        """_ERROR_STATE stays False when model.nc matches config num_classes."""
        inf = _load_inference_module(nc_model=5, nc_config=5)
        (tmp_path / "best.pt").write_bytes(b"fake")
        (tmp_path / "code").mkdir()
        (tmp_path / "code" / "model_config.json").write_text(
            json.dumps({"num_classes": 5, "class_names": ["a"] * 5, "input_size": 416})
        )
        inf.model_fn(str(tmp_path))
        assert inf._ERROR_STATE is False

    def test_error_state_true_on_class_count_mismatch(self, tmp_path):
        """Req 9.4: _ERROR_STATE is set True when model nc != config num_classes."""
        inf = _load_inference_module(nc_model=3, nc_config=5)
        (tmp_path / "best.pt").write_bytes(b"fake")
        (tmp_path / "code").mkdir()
        (tmp_path / "code" / "model_config.json").write_text(
            json.dumps({"num_classes": 5, "class_names": ["a"] * 5, "input_size": 416})
        )
        inf.model_fn(str(tmp_path))
        assert inf._ERROR_STATE is True

    def test_mismatch_logs_structured_json_error(self, tmp_path, capsys):
        """Req 9.4: structured JSON error with required fields is printed on mismatch."""
        inf = _load_inference_module(nc_model=3, nc_config=5)
        (tmp_path / "best.pt").write_bytes(b"fake")
        (tmp_path / "code").mkdir()
        (tmp_path / "code" / "model_config.json").write_text(
            json.dumps({"num_classes": 5, "class_names": ["a"] * 5, "input_size": 416})
        )
        inf.model_fn(str(tmp_path))
        captured = capsys.readouterr()
        logged = json.loads(captured.out.strip())
        assert logged["error_type"] == "ClassCountMismatch"
        assert logged["model_class_count"] == 3
        assert logged["expected_num_classes"] == 5
        assert "model_config_path" in logged

    def test_mismatch_log_level_is_error(self, tmp_path, capsys):
        inf = _load_inference_module(nc_model=2, nc_config=4)
        (tmp_path / "best.pt").write_bytes(b"fake")
        (tmp_path / "code").mkdir()
        (tmp_path / "code" / "model_config.json").write_text(
            json.dumps({"num_classes": 4, "class_names": ["a"] * 4, "input_size": 416})
        )
        inf.model_fn(str(tmp_path))
        captured = capsys.readouterr()
        logged = json.loads(captured.out.strip())
        assert logged["level"] == "ERROR"


# ---------------------------------------------------------------------------
# input_fn
# ---------------------------------------------------------------------------


class TestInputFn:
    """Tests for input_fn (Req 5.7, 5.8)."""

    def _jpeg_bytes(self) -> bytes:
        """Return a minimal valid JPEG byte sequence."""
        buf = io.BytesIO()
        try:
            from PIL import Image as _PILImage
            img = _PILImage.new("RGB", (10, 10), color=(128, 0, 0))
            img.save(buf, format="JPEG")
            return buf.getvalue()
        except ImportError:
            # Fallback: use JPEG magic bytes (won't parse, but PIL stub doesn't care)
            return b"\xff\xd8\xff\xe0" + b"\x00" * 100

    def _png_bytes(self) -> bytes:
        buf = io.BytesIO()
        try:
            from PIL import Image as _PILImage
            img = _PILImage.new("RGB", (10, 10), color=(0, 128, 0))
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    def test_accepts_jpeg_content_type(self):
        inf = _load_inference_module()
        result = inf.input_fn(self._jpeg_bytes(), "image/jpeg")
        assert result is not None

    def test_accepts_png_content_type(self):
        inf = _load_inference_module()
        result = inf.input_fn(self._png_bytes(), "image/png")
        assert result is not None

    def test_raises_value_error_for_unsupported_content_type(self):
        """Req 5.8: ValueError raised for any content type other than JPEG/PNG."""
        inf = _load_inference_module()
        with pytest.raises(ValueError, match="Unsupported content type"):
            inf.input_fn(b"data", "image/gif")

    def test_error_message_includes_bad_content_type(self):
        inf = _load_inference_module()
        with pytest.raises(ValueError) as exc_info:
            inf.input_fn(b"data", "application/octet-stream")
        assert "application/octet-stream" in str(exc_info.value)

    def test_raises_for_text_plain_content_type(self):
        inf = _load_inference_module()
        with pytest.raises(ValueError):
            inf.input_fn(b"hello", "text/plain")

    def test_raises_for_empty_content_type(self):
        inf = _load_inference_module()
        with pytest.raises(ValueError):
            inf.input_fn(b"data", "")


# ---------------------------------------------------------------------------
# predict_fn
# ---------------------------------------------------------------------------


class TestPredictFn:
    """Tests for predict_fn (Req 5.9 partial)."""

    def test_returns_dict(self):
        inf = _load_inference_module()
        model = MagicMock()
        model.return_value = []
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        result = inf.predict_fn(tensor, model)
        assert isinstance(result, dict)

    def test_result_contains_inference_time_ms(self):
        inf = _load_inference_module()
        model = MagicMock(return_value=[])
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        result = inf.predict_fn(tensor, model)
        assert "inference_time_ms" in result

    def test_inference_time_ms_is_positive(self):
        inf = _load_inference_module()
        model = MagicMock(return_value=[])
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        result = inf.predict_fn(tensor, model)
        assert result["inference_time_ms"] >= 0

    def test_result_contains_raw_results(self):
        inf = _load_inference_module()
        sentinel = object()
        model = MagicMock(return_value=sentinel)
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        result = inf.predict_fn(tensor, model)
        assert "results" in result
        assert result["results"] is sentinel

    def test_result_contains_input_shape(self):
        inf = _load_inference_module()
        model = MagicMock(return_value=[])
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        result = inf.predict_fn(tensor, model)
        assert "input_shape" in result
        assert result["input_shape"] == [1, 3, 416, 416]

    def test_calls_model_with_tensor(self):
        inf = _load_inference_module()
        model = MagicMock(return_value=[])
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        inf.predict_fn(tensor, model)
        model.assert_called_once_with(tensor)

    def test_exceptions_propagate_to_caller(self):
        """predict_fn exceptions must propagate so output_fn can catch them."""
        inf = _load_inference_module()
        model = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        tensor = MagicMock()
        tensor.shape = [1, 3, 416, 416]
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            inf.predict_fn(tensor, model)


# ---------------------------------------------------------------------------
# output_fn
# ---------------------------------------------------------------------------


class TestOutputFn:
    """Tests for output_fn (Req 5.9, 5.11, 8.2, 9.4)."""

    def _make_prediction(self, inf_module):
        """Build a minimal prediction dict that output_fn can serialise."""
        return {
            "results": [],
            "inference_time_ms": 5.0,
            "input_shape": [1, 3, 416, 416],
        }

    def test_returns_tuple(self):
        inf = _load_inference_module()
        pred = self._make_prediction(inf)
        result = inf.output_fn(pred, "application/json")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_second_element_is_application_json(self):
        """Req 5.9: content type must be 'application/json'."""
        inf = _load_inference_module()
        pred = self._make_prediction(inf)
        _, content_type = inf.output_fn(pred, "application/json")
        assert content_type == "application/json"

    def test_first_element_is_valid_json(self):
        inf = _load_inference_module()
        pred = self._make_prediction(inf)
        json_str, _ = inf.output_fn(pred, "application/json")
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_response_contains_detections_key(self):
        """Req 5.9: response must have 'detections' list."""
        inf = _load_inference_module()
        pred = self._make_prediction(inf)
        json_str, _ = inf.output_fn(pred, "application/json")
        parsed = json.loads(json_str)
        assert "detections" in parsed
        assert isinstance(parsed["detections"], list)

    def test_response_contains_inference_time_ms(self):
        """Req 5.9: response must have inference_time_ms > 0."""
        inf = _load_inference_module()
        pred = self._make_prediction(inf)
        json_str, _ = inf.output_fn(pred, "application/json")
        parsed = json.loads(json_str)
        assert "inference_time_ms" in parsed
        assert parsed["inference_time_ms"] > 0

    def test_response_contains_model_run_name(self):
        """Req 5.9: response must have model_run_name."""
        inf = _load_inference_module()
        pred = self._make_prediction(inf)
        json_str, _ = inf.output_fn(pred, "application/json")
        parsed = json.loads(json_str)
        assert "model_run_name" in parsed

    def test_error_state_returns_http_500_body(self):
        """Req 9.4: when _ERROR_STATE is True, output_fn returns an error JSON."""
        inf = _load_inference_module()
        inf._ERROR_STATE = True
        pred = self._make_prediction(inf)
        json_str, _ = inf.output_fn(pred, "application/json")
        parsed = json.loads(json_str)
        assert "error" in parsed

    def test_error_state_still_returns_application_json(self):
        inf = _load_inference_module()
        inf._ERROR_STATE = True
        pred = self._make_prediction(inf)
        _, content_type = inf.output_fn(pred, "application/json")
        assert content_type == "application/json"

    def test_error_state_message_mentions_redeploy(self):
        """Req 9.4: error message should direct operator to redeploy."""
        inf = _load_inference_module()
        inf._ERROR_STATE = True
        pred = self._make_prediction(inf)
        json_str, _ = inf.output_fn(pred, "application/json")
        assert "redeploy" in json_str.lower() or "Redeploy" in json_str

    def test_predict_fn_exception_caught_by_output_fn(self):
        """Req 5.11: exceptions from predict_fn are caught; no re-raise."""
        inf = _load_inference_module()
        broken_pred = {
            "results": None,  # will cause AttributeError when iterated
            "inference_time_ms": 5.0,
            "input_shape": [1, 3, 416, 416],
        }
        # Should NOT raise
        json_str, content_type = inf.output_fn(broken_pred, "application/json")
        assert content_type == "application/json"
        parsed = json.loads(json_str)
        assert "error" in parsed

    def test_exception_is_logged_as_structured_json(self, capsys):
        """Req 5.11 / 8.2: structured JSON log entry is printed on exception."""
        inf = _load_inference_module()
        broken_pred = {"results": None, "inference_time_ms": 5.0, "input_shape": [1, 3, 416, 416]}
        inf.output_fn(broken_pred, "application/json")
        captured = capsys.readouterr()
        # stdout should contain a JSON log entry
        log_entry = json.loads(captured.out.strip())
        assert log_entry["level"] == "ERROR"
        assert "error_type" in log_entry
        assert "error_message" in log_entry
        assert "timestamp" in log_entry

    def test_exception_log_contains_input_shape(self, capsys):
        """Req 8.2: log entry must include input_shape."""
        inf = _load_inference_module()
        broken_pred = {"results": None, "inference_time_ms": 5.0, "input_shape": [1, 3, 416, 416]}
        inf.output_fn(broken_pred, "application/json")
        captured = capsys.readouterr()
        log_entry = json.loads(captured.out.strip())
        assert "input_shape" in log_entry

    def test_secondary_log_exception_does_not_propagate(self):
        """Req 5.11 / 8.2: if logging itself fails, HTTP 500 is still returned."""
        inf = _load_inference_module()
        broken_pred = {"results": None, "inference_time_ms": 5.0, "input_shape": [1, 3, 416, 416]}

        # Make json.dumps raise inside the log path
        import builtins
        real_print = builtins.print
        call_count = {"n": 0}

        def failing_print(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("log write failed")
            return real_print(*args, **kwargs)

        with patch("builtins.print", side_effect=failing_print):
            # Should still return a tuple without raising
            result = inf.output_fn(broken_pred, "application/json")

        assert isinstance(result, tuple)
        assert len(result) == 2
