import pandas as pd
from pathlib import Path

# -----------------------------
# Locate the CSV file
# -----------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "resume_jd_data.csv"


def load_data(path: Path) -> pd.DataFrame:
    """
    Load resume–job description dataset from CSV.
    """
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Basic validation checks on the dataset.
    """
    required_columns = {
        "resume_text",
        "job_description_text",
        "label"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    print("Data loaded successfully")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    validate_data(df)
    print(df.head(3))
