# src/train.py
import os
import yaml
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    model_type = params["model"]["type"]

    train_path = "data/processed/train.csv"
    df = pd.read_csv(train_path)
    if "Outcome" not in df.columns:
        raise ValueError("Outcome column missing in train.csv")

    X_train = df.drop(columns=["Outcome"])
    y_train = df["Outcome"]

    if model_type == "DecisionTreeClassifier":
        mparams = params["model"]["DecisionTreeClassifier"]
        model = DecisionTreeClassifier(
            criterion=mparams.get("criterion", "gini"),
            max_depth=mparams.get("max_depth", None),
            min_samples_split=mparams.get("min_samples_split", 2),
            random_state=params["dataset"]["random_state"]
        )
    elif model_type == "RandomForestClassifier":
        mparams = params["model"]["RandomForestClassifier"]
        model = RandomForestClassifier(
            n_estimators=mparams.get("n_estimators", 100),
            max_depth=mparams.get("max_depth", None),
            min_samples_split=mparams.get("min_samples_split", 2),
            n_jobs=mparams.get("n_jobs", -1),
            random_state=params["dataset"]["random_state"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Training complete → models/model.pkl ({model_type})")

if __name__ == "__main__":
    main()
