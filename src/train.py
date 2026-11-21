"""Обучение модели RandomForest на датасете Obesity Prediction.

Сохраняет модель, энкодер и список признаков в artifacts/.
"""
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ARTIFACTS_DIR = "artifacts"
DATASET_PATH = os.path.join(ARTIFACTS_DIR, "obesity_dataset.csv")
TARGET_COLUMN = "NObeyesdad"


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Датасет не найден: {path}. Сначала запустите src/download_data.py"
        )
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """Кодирует признаки и целевую переменную."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def train_model(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, y_test, predictions


def save_artifacts(model, label_encoder, feature_columns) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.joblib"))
    joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))


def main() -> None:
    df = load_data()
    X, y_encoded, label_encoder = prepare_features(df)
    model, accuracy, y_test, predictions = train_model(X, y_encoded)

    print(f"Accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test, predictions, target_names=label_encoder.classes_
        )
    )

    save_artifacts(model, label_encoder, X.columns.tolist())
    print("Артефакты сохранены в artifacts/: model.joblib, label_encoder.joblib, feature_columns.joblib")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка при обучении: {exc}", file=sys.stderr)
        sys.exit(1)