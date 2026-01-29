import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Paths
# -----------------------------
EMBEDDINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "embeddings.npz"
SIMILARITY_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "similarity_scores.csv"


if __name__ == "__main__":
    # load embeddings
    data = np.load(EMBEDDINGS_PATH)

    resume_embeddings = data["resume_embeddings"]
    jd_embeddings = data["jd_embeddings"]
    labels = data["labels"]

    # calculate cosine similarity
    similarity_scores = cosine_similarity(resume_embeddings, jd_embeddings).diagonal()

    # save results
    import pandas as pd
    df_sim = pd.DataFrame({
        "similarity_score": similarity_scores,
        "label": labels
    })

    df_sim.to_csv(SIMILARITY_PATH, index=False)

    print("Step 5 completed: Similarity calculated")
    print(df_sim.head())
