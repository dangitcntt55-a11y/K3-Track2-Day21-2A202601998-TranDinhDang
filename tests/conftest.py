import os

# Set MLflow tracking URI to use SQLite backend
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
