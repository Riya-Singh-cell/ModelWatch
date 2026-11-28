import pandas as pd
from io import BytesIO


def read_csv_upload(filename: str, content: bytes) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")
    if not content:
        raise ValueError("The uploaded dataset is empty.")
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Invalid CSV file: {exc}") from exc
    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid = [column for column in required if df[column].isna().all()]
    if invalid:
        raise ValueError(f"Columns contain no valid numeric values: {', '.join(invalid)}")
    return required
