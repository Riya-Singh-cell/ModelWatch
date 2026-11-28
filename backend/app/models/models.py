from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class MonitoredModel(Base):
    __tablename__ = "monitored_models"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    description = Column(Text, default="")
    target_column = Column(String, nullable=False)
    feature_columns = Column(JSON, nullable=False, default=list)
    status = Column(String, default="Not monitored")
    threshold_config = Column(JSON, default=dict)
    artifact_path = Column(String, nullable=True)
    original_model_filename = Column(String, nullable=True)
    model_class = Column(String, nullable=True)
    model_feature_names = Column(JSON, nullable=True)
    model_feature_count = Column(Integer, nullable=True)
    trusted_model = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    datasets = relationship("Dataset", back_populates="model", cascade="all, delete-orphan")
    runs = relationship("MonitoringRun", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("monitored_models.id"), nullable=False)
    dataset_type = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    columns = Column(JSON, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    model = relationship("MonitoredModel", back_populates="datasets")


class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("monitored_models.id"), nullable=False)
    reference_dataset_id = Column(Integer, ForeignKey("datasets.id"))
    production_dataset_id = Column(Integer, ForeignKey("datasets.id"))
    health_status = Column(String, nullable=False)
    health_explanation = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    model = relationship("MonitoredModel", back_populates="runs")
    metrics = relationship("PerformanceMetric", back_populates="run", cascade="all, delete-orphan")
    drift = relationship("DriftResult", back_populates="run", uselist=False, cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="run", cascade="all, delete-orphan")
    analyses = relationship("AgentAnalysis", back_populates="run", cascade="all, delete-orphan")


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("monitoring_runs.id"), nullable=False)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=True)
    previous_value = Column(Float, nullable=True)
    run = relationship("MonitoringRun", back_populates="metrics")


class DriftResult(Base):
    __tablename__ = "drift_results"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("monitoring_runs.id"), nullable=False)
    overall_drift = Column(Boolean, default=False)
    drifted_features = Column(Integer, default=0)
    summary = Column(Text, default="")
    run = relationship("MonitoringRun", back_populates="drift")
    features = relationship("FeatureDrift", back_populates="drift", cascade="all, delete-orphan")


class FeatureDrift(Base):
    __tablename__ = "feature_drift"
    id = Column(Integer, primary_key=True)
    drift_result_id = Column(Integer, ForeignKey("drift_results.id"), nullable=False)
    feature = Column(String, nullable=False)
    ks_statistic = Column(Float, nullable=True)
    ks_p_value = Column(Float, nullable=True)
    psi = Column(Float, nullable=True)
    drift_detected = Column(Boolean, default=False)
    severity = Column(String, default="None")
    explanation = Column(Text, default="")
    drift = relationship("DriftResult", back_populates="features")


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("monitored_models.id"), nullable=False)
    run_id = Column(Integer, ForeignKey("monitoring_runs.id"), nullable=True)
    severity = Column(String, nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    model = relationship("MonitoredModel", back_populates="alerts")
    run = relationship("MonitoringRun", back_populates="alerts")


class AgentAnalysis(Base):
    __tablename__ = "agent_analyses"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("monitoring_runs.id"), nullable=False)
    agent_name = Column(String, nullable=False)
    output = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("MonitoringRun", back_populates="analyses")
