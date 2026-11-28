import logging
import os
import uuid
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)
MODEL_STORAGE = Path(os.environ.get(
    "MODELWATCH_MODEL_STORAGE",
    str(Path(__file__).resolve().parents[2] / "storage" / "models"),
)).resolve()
MAX_MODEL_BYTES = 100 * 1024 * 1024


def save_trusted_model(filename: str, content: bytes):
    if not filename or Path(filename).suffix.lower() != ".pkl":
        raise ValueError("Only .pkl model files are supported.")
    if not content:
        raise ValueError("The uploaded model file is empty.")
    if len(content) > MAX_MODEL_BYTES:
        raise ValueError("The model file exceeds the 100 MB upload limit.")

    MODEL_STORAGE.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_STORAGE / f"{uuid.uuid4().hex}.pkl"
    artifact_path.write_bytes(content)
    try:
        model = joblib.load(artifact_path)
        if not callable(getattr(model, "predict", None)):
            raise ValueError("The uploaded object does not expose a predict method.")
        feature_names = getattr(model, "feature_names_in_", None)
        feature_count = getattr(model, "n_features_in_", None)
        metadata = {
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "feature_names": [str(name) for name in feature_names] if feature_names is not None else None,
            "feature_count": int(feature_count) if feature_count is not None else None,
            "task_type": "regression" if getattr(model, "_estimator_type", None) == "regressor" else "classification",
        }
    except Exception as exc:
        artifact_path.unlink(missing_ok=True)
        logger.exception("Could not validate trusted uploaded model artifact")
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Could not load the trusted .pkl model file.") from exc
    return artifact_path, metadata


def load_trusted_model(artifact_path: str):
    path = Path(artifact_path).resolve()
    if not path.is_relative_to(MODEL_STORAGE):
        raise ValueError("Model artifact path is outside private model storage.")
    if not path.is_file():
        raise ValueError("The stored model artifact is missing.")
    return joblib.load(path)


def remove_model_artifact(artifact_path: str | None):
    if not artifact_path:
        return
    path = Path(artifact_path).resolve()
    if path.is_relative_to(MODEL_STORAGE) and path.is_file():
        path.unlink()
