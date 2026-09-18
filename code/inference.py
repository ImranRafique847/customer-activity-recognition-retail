"""
SageMaker PyTorch inference handler for the retail CV YOLO model.

SageMaker calls these four entry points in order:
    model_fn  → input_fn  → predict_fn  → output_fn

COST NOTE: If using the real-time endpoint, delete it after 30 minutes of
idle time (zero invocations) to avoid unnecessary charges.
"""

import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_ERROR_STATE: bool = False   # set True by model_fn on class-count mismatch
_MODEL_RUN_NAME: str = "unknown"

# ---------------------------------------------------------------------------
# model_fn
# ---------------------------------------------------------------------------


def model_fn(model_dir: str) -> YOLO:
    """Load best.pt and validate class count against model_config.json.

    Loads best.pt from model_dir using ultralytics.YOLO(model_path).
    Reads model_config.json from model_dir and validates that
    model.model.nc == config['num_classes'].

    On mismatch: logs a structured JSON error with model_class_count,
    expected_num_classes, and model_config_path; sets _ERROR_STATE = True.
    All subsequent inference calls will return HTTP 500 until the endpoint
    is redeployed.

    Returns the YOLO model object (callable).
    """
    global _ERROR_STATE, _MODEL_RUN_NAME

    model_path = os.path.join(model_dir, "best.pt")
    # SageMaker extracts model.tar.gz to model_dir; model_config.json is
    # bundled at code/model_config.json inside the archive.
    config_path = os.path.join(model_dir, "code", "model_config.json")

    model = YOLO(model_path)

    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        _log_error(
            error_type="ConfigLoadError",
            error_message=str(exc),
            input_shape=[],
            extra={"model_config_path": config_path},
        )
        _ERROR_STATE = True
        return model

    _MODEL_RUN_NAME = config.get("model_run_name", "unknown")
    expected_nc = config.get("num_classes", -1)

    try:
        model_nc = model.model.nc
    except AttributeError:
        model_nc = -1

    if model_nc != expected_nc:
        # Log the exact structured JSON required by Req 5.6 / 9.4
        mismatch_entry = {
            "level": "ERROR",
            "error_type": "ClassCountMismatch",
            "model_class_count": model_nc,
            "expected_num_classes": expected_nc,
            "model_config_path": config_path,
        }
        try:
            print(json.dumps(mismatch_entry))
        except Exception:
            pass
        _ERROR_STATE = True

    return model


# ---------------------------------------------------------------------------
# input_fn
# ---------------------------------------------------------------------------


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    """Decode a JPEG or PNG request body into a normalised float32 tensor.

    Returns a tensor of shape [1, 3, 416, 416] with values in [0.0, 1.0].
    Raises ValueError for unsupported content types.
    """
    if content_type not in {"image/jpeg", "image/png"}:
        raise ValueError(f"Unsupported content type: {content_type}")

    img = Image.open(io.BytesIO(request_body)).convert("RGB").resize((416, 416))
    arr = np.array(img, dtype=np.float32) / 255.0          # (416, 416, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 416, 416)
    return tensor


# ---------------------------------------------------------------------------
# predict_fn
# ---------------------------------------------------------------------------


def predict_fn(input_tensor: torch.Tensor, model: YOLO) -> dict:
    """Run the YOLO model and return raw results plus timing."""
    start = time.time()
    results = model(input_tensor)
    inference_time_ms = (time.time() - start) * 1000.0
    return {
        "results": results,
        "inference_time_ms": inference_time_ms,
        "input_shape": list(input_tensor.shape),
    }


# ---------------------------------------------------------------------------
# output_fn
# ---------------------------------------------------------------------------


def output_fn(prediction, accept: str) -> tuple:
    """Serialise YOLO predictions to JSON.

    Returns (json_string, 'application/json').
    On any exception: logs a structured error entry and returns HTTP 500.
    """
    if _ERROR_STATE:
        return (
            json.dumps({
                "error": (
                    "Model class count mismatch. "
                    "Redeploy endpoint with a consistent model and config."
                )
            }),
            "application/json",
        )

    try:
        results = prediction["results"]
        inference_time_ms = prediction["inference_time_ms"]
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxyn = box.xyxyn[0].tolist()   # normalized [x_min, y_min, x_max, y_max]
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = result.names[cls] if result.names else str(cls)
                detections.append({
                    "label": label,
                    "confidence": round(conf, 6),
                    "bbox": [round(v, 6) for v in xyxyn],
                })

        response = {
            "detections": detections,
            "inference_time_ms": round(inference_time_ms, 3),
            "model_run_name": _MODEL_RUN_NAME,
        }
        return (json.dumps(response), "application/json")

    except Exception as exc:
        try:
            log_entry = {
                "level": "ERROR",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "input_shape": prediction.get("input_shape", []),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            print(json.dumps(log_entry))
        except Exception:
            pass  # never let a logging failure propagate

        return (json.dumps({"error": str(exc)}), "application/json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_error(
    error_type: str,
    error_message: str,
    input_shape: list,
    extra: dict = None,
) -> None:
    """Print a structured JSON error entry to stdout."""
    entry = {
        "level": "ERROR",
        "error_type": error_type,
        "error_message": error_message,
        "input_shape": input_shape,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    if extra:
        entry.update(extra)
    try:
        print(json.dumps(entry))
    except Exception:
        pass
