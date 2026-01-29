import pandas as pd
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "resume_jd_data.csv"
PROCESSED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_data.csv"
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)



def clean_text(text: str) -> str:
    """
    Clean resume or job description text.
    """
    text = text.lower()
    text = re.sub(r"\S+@\S+", "", text)      # remove emails
    text = re.sub(r"\d{10}", "", text)       # remove phone numbers
    text = re.sub(r"[^a-z\s]", " ", text)   # remove special characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to resume and job description columns.
    """
    df = df.copy()
    df["resume_clean"] = df["resume_text"].apply(clean_text)
    df["jd_clean"] = df["job_description_text"].apply(clean_text)
    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    df_clean = clean_dataset(df)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Text cleaning completed")
    print(df_clean[["resume_clean", "jd_clean"]].head(3))
