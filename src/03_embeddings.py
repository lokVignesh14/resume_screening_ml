import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
CLEAN_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_data.csv"
EMBEDDINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "embeddings.npz"

# ensure directory exists
EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(text_list):
    """
    Convert list of texts into embeddings.
    """
    embeddings = model.encode(
        text_list,
        show_progress_bar=True
    )
    return embeddings


if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv(CLEAN_DATA_PATH)

    # Generate embeddings
    resume_embeddings = generate_embeddings(df["resume_clean"].tolist())
    jd_embeddings = generate_embeddings(df["jd_clean"].tolist())

    # Save embeddings
    np.savez(
        EMBEDDINGS_PATH,
        resume_embeddings=resume_embeddings,
        jd_embeddings=jd_embeddings,
        labels=df["label"].values
    )

    print("Embeddings generated and saved successfully")
    print("Resume embedding shape:", resume_embeddings.shape)
