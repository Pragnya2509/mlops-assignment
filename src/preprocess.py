import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Load dataset
    df = pd.read_csv(params["dataset"]["path"])

    # Split into features and target
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["dataset"]["test_size"],
        random_state=params["dataset"]["random_state"]
    )

    # Handle missing values
    fill_strategy = params["preprocessing"]["fill_missing"]
    if fill_strategy == "median":
        X_train = X_train.fillna(X_train.median(numeric_only=True))
        X_test = X_test.fillna(X_train.median(numeric_only=True))
    elif fill_strategy == "mean":
        X_train = X_train.fillna(X_train.mean(numeric_only=True))
        X_test = X_test.fillna(X_train.mean(numeric_only=True))
    elif fill_strategy == "zero":
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

    # Scaling
    if params["preprocessing"]["scale"]:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns, index=X_test.index)

    # Reattach target column
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("✅ Preprocessing complete → data/processed/train.csv, data/processed/test.csv")

if __name__ == "__main__":
    main()
