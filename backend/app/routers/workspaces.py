import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import Dataset, MonitoredModel
from app.services.model_artifacts import MAX_MODEL_BYTES, save_trusted_model, remove_model_artifact

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model workspace"])
PREVIEW_ROWS = 8


@router.post("/model-workspaces", status_code=201)
async def upload_model(
    file: UploadFile = File(...),
    trusted_source: bool = Form(...),
    db: Session = Depends(get_db),
):
    if not trusted_source:
        raise HTTPException(
            422,
            "Pickle files can execute code when loaded. Confirm the model comes from a trusted source.",
        )
    content = await file.read(MAX_MODEL_BYTES + 1)
    if len(content) > MAX_MODEL_BYTES:
        raise HTTPException(413, "The model file exceeds the 100 MB upload limit.")
    try:
        artifact_path, metadata = save_trusted_model(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    model = MonitoredModel(
        name=Path(file.filename or "uploaded-model.pkl").stem,
        model_type=metadata["task_type"],
        description="Uploaded trained model",
        target_column="",
        feature_columns=[],
        artifact_path=str(artifact_path),
        original_model_filename=Path(file.filename or "uploaded-model.pkl").name,
        model_class=metadata["model_class"],
        model_feature_names=metadata["feature_names"],
        model_feature_count=metadata["feature_count"],
        trusted_model=True,
    )
    try:
        db.add(model)
        db.commit()
        db.refresh(model)
    except Exception:
        db.rollback()
        remove_model_artifact(str(artifact_path))
        logger.exception("Could not save uploaded model metadata")
        raise HTTPException(500, "Could not save model metadata.")
    return {
        "id": model.id,
        "filename": model.original_model_filename,
        "model_class": model.model_class,
        "feature_names": model.model_feature_names,
        "feature_count": model.model_feature_count,
        "trusted_source_confirmed": model.trusted_model,
    }


@router.get("/models/{model_id}/datasets")
def workspace_datasets(model_id: int, db: Session = Depends(get_db)):
    model = db.get(MonitoredModel, model_id)
    if not model or not model.artifact_path:
        raise HTTPException(404, "Uploaded model workspace not found.")
    datasets = db.query(Dataset).filter_by(model_id=model_id).order_by(Dataset.id.asc()).all()
    result = []
    for dataset in datasets:
        frame = pd.read_csv(pd.io.common.StringIO(dataset.content))
        result.append({
            "id": dataset.id,
            "dataset_type": dataset.dataset_type,
            "filename": dataset.filename,
            "row_count": dataset.row_count,
            "column_count": len(dataset.columns),
            "columns": dataset.columns,
            "preview": frame.head(PREVIEW_ROWS).where(pd.notna(frame), None).to_dict(orient="records"),
        })
    return result
