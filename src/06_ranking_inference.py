import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Paths
# -----------------------------
CLEAN_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_data.csv"

# -----------------------------
# Load model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


def rank_resumes(job_description: str, resumes: list, top_k: int = 3):
    """
    Rank resumes based on similarity to a job description.
    """
    jd_embedding = model.encode([job_description])
    resume_embeddings = model.encode(resumes)

    similarities = cosine_similarity(resume_embeddings, jd_embedding).flatten()

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "resume_text": resumes[idx],
            "similarity_score": round(similarities[idx], 3)
        })

    return results


if __name__ == "__main__":
    # load cleaned data
    df = pd.read_csv(CLEAN_DATA_PATH)

    resumes = df["resume_clean"].tolist()

    # example new job description
    new_jd = "Looking for a machine learning engineer with Python and NLP experience"

    ranked_results = rank_resumes(new_jd, resumes, top_k=3)

    print("Top matching resumes:\n")
    for rank, item in enumerate(ranked_results, start=1):
        print(f"Rank {rank}")
        print("Score:", item["similarity_score"])
        print("Resume:", item["resume_text"][:120], "\n")
