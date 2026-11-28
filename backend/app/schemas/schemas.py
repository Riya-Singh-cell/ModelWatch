from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    model_type: str = "classification"
    description: str = ""
    target_column: str = Field(min_length=1)
    feature_columns: List[str]
    threshold_config: Dict[str, float] = Field(default_factory=dict)


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    model_type: str
    description: str
    target_column: str
    feature_columns: List[str]
    status: str
    threshold_config: Dict[str, Any]
    original_model_filename: Optional[str] = None
    model_class: Optional[str] = None
    model_feature_names: Optional[List[str]] = None
    model_feature_count: Optional[int] = None
    trusted_model: bool = False
    health_explanation: Optional[str] = None
    latest_run_id: Optional[int] = None
    last_monitored_at: Optional[str] = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    dataset_type: str
    filename: str
    row_count: int
    columns: List[str]


class MonitoringRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    health_status: str
    health_explanation: str
