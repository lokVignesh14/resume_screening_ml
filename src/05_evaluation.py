import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# -----------------------------
# Path
# -----------------------------
SIMILARITY_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "similarity_scores.csv"


def evaluate_similarity(df: pd.DataFrame, threshold: float = 0.5):
    """
    Evaluate similarity scores against labels using basic metrics.
    """
    y_true = df["label"]
    y_scores = df["similarity_score"]

    # binary prediction using threshold
    y_pred = (y_scores >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return auc, precision, recall


if __name__ == "__main__":
    df = pd.read_csv(SIMILARITY_PATH)

    auc, precision, recall = evaluate_similarity(df)

    print("Step 6 completed: Evaluation results")
    print("AUC:", round(auc, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
