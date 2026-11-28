from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "modelwatch.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def initialize_database():
    Base.metadata.create_all(bind=engine)
    columns = {column["name"] for column in inspect(engine).get_columns("monitored_models")}
    migrations = {
        "artifact_path": "ALTER TABLE monitored_models ADD COLUMN artifact_path VARCHAR",
        "original_model_filename": "ALTER TABLE monitored_models ADD COLUMN original_model_filename VARCHAR",
        "model_class": "ALTER TABLE monitored_models ADD COLUMN model_class VARCHAR",
        "model_feature_names": "ALTER TABLE monitored_models ADD COLUMN model_feature_names JSON",
        "model_feature_count": "ALTER TABLE monitored_models ADD COLUMN model_feature_count INTEGER",
        "trusted_model": "ALTER TABLE monitored_models ADD COLUMN trusted_model BOOLEAN NOT NULL DEFAULT 0",
    }
    with engine.begin() as connection:
        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(text(statement))
        connection.execute(text(
            "UPDATE monitored_models SET status = 'Not monitored' "
            "WHERE NOT EXISTS (SELECT 1 FROM monitoring_runs "
            "WHERE monitoring_runs.model_id = monitored_models.id)"
        ))
        connection.execute(text(
            "UPDATE monitored_models SET model_type = 'regression' "
            "WHERE model_class LIKE '%LinearRegression' "
            "OR model_class LIKE '%Regressor' "
            "OR model_class LIKE '%SVR'"
        ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
