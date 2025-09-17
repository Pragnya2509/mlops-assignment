# src/evaluate.py
import os
import json
import pandas as pd
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_float(x):
    try:
        return None if pd.isna(x) else float(x)
    except Exception:
        return None

def main():
    os.makedirs("experiments", exist_ok=True)
    model = load_model("models/model.pkl")

    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }

    if y_proba is not None and len(set(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["roc_auc"] = None

    # --- DVC metrics (simple numeric object) ---
    metrics_for_dvc = {k: v for k, v in metrics.items() if k != "timestamp"}
    with open("experiments/metrics.json", "w") as f:
        json.dump(metrics_for_dvc, f, indent=2)

    # --- Human-friendly JSON (optional) ---
    with open("experiments/results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Append to history CSV ---
    hist_path = "experiments/all_results.csv"
    hist_df = pd.DataFrame([metrics])
    hist_df.to_csv(hist_path, mode="a", index=False, header=not os.path.exists(hist_path))

    # --- Create plot-friendly JSON from history (array of records) ---
    try:
        df = pd.read_csv(hist_path)
        # ensure consistent columns and proper numeric dtype
        cols = ["timestamp", "accuracy", "precision", "recall", "f1", "roc_auc"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        plot_df = df[cols].copy()
        # convert numeric columns safely
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
        records = []
        for _, row in plot_df.iterrows():
            rec = {"timestamp": str(row["timestamp"])}
            for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                val = row[c]
                rec[c] = None if pd.isna(val) else float(val)
            records.append(rec)
        with open("experiments/plots.json", "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        # fallback: write single-record plots.json
        with open("experiments/plots.json", "w") as f:
            json.dump([metrics], f, indent=2)

    print("✅ Evaluate complete → experiments/metrics.json + experiments/results.json + experiments/all_results.csv + experiments/plots.json")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
