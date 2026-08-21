import os

# Set MLflow tracking URI before importing mlflow
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    use_combined_data: bool = False,
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params             : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path          : duong dan den file du lieu huan luyen.
        eval_path          : duong dan den file du lieu danh gia.
        use_combined_data  : neu True, load them train_phase2.csv de tang dung luong training

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)

    # Use combined data for better accuracy
    if use_combined_data:
        try:
            df_train2 = pd.read_csv("data/train_phase2.csv")
            df_train = pd.concat([df_train, df_train2], ignore_index=True)
            print(f"Using combined data: {len(df_train)} samples")
        except FileNotFoundError:
            print("train_phase2.csv not found, using only train_phase1.csv")

    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        # Log all parameters
        mlflow.log_params(params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("eval_samples", len(X_eval))
        mlflow.log_param("use_combined_data", use_combined_data)

        # Create best ensemble model
        ensemble = VotingClassifier(
            estimators=[
                ('rf1', RandomForestClassifier(n_estimators=1000, max_depth=25, bootstrap=False, random_state=42, n_jobs=-1)),
                ('rf2', RandomForestClassifier(n_estimators=1000, max_depth=30, bootstrap=False, random_state=43, n_jobs=-1)),
                ('et1', ExtraTreesClassifier(n_estimators=500, max_depth=None, random_state=44, n_jobs=-1)),
                ('hgb', HistGradientBoostingClassifier(max_iter=300, max_depth=12, random_state=46)),
            ],
            voting='soft',
            n_jobs=-1
        )

        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_eval)

        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model using pickle format (compatible with all sklearn models)
        mlflow.sklearn.log_model(
            ensemble,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(ensemble, "models/model.pkl")

    return float(acc)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Use combined data for better accuracy
    acc = train(params, use_combined_data=True)

    print(f"\nFinal accuracy: {acc:.4f}")
    if acc >= 0.70:
        print("✅ Model passed eval threshold (>= 0.70)")
    else:
        print(f"⚠️ Model below eval threshold: {acc:.4f} < 0.70")
