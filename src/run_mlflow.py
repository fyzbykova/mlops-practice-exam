"""Логирование обучения модели в MLflow (SQLite backend).

Запускает отдельный эксперимент, используя те же данные и гиперпараметры,
что и src/train.py.
"""
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ARTIFACTS_DIR = "artifacts"
DATASET_PATH = os.path.join(ARTIFACTS_DIR, "obesity_dataset.csv")
TARGET_COLUMN = "NObeyesdad"
MLFLOW_DB = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Obesity_Classification"


def load_and_prepare(path: str = DATASET_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def main() -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}. Сначала запустите src/download_data.py"
        )

    X, y_encoded, _ = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RF_baseline"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print("Эксперимент залогирован в MLflow (SQLite)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка при логировании в MLflow: {exc}", file=sys.stderr)
        sys.exit(1)