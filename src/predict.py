"""Локальный инференс: загружает модель из artifacts/ и делает предсказание.

Пример запуска:
    python src/predict.py
"""
import os
import sys

import joblib
import pandas as pd

ARTIFACTS_DIR = "artifacts"


def load_artifacts():
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")
    columns_path = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

    for path in (model_path, encoder_path, columns_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Артефакт не найден: {path}. Сначала запустите src/train.py"
            )

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    feature_columns = joblib.load(columns_path)

    return model, label_encoder, feature_columns


def predict_one(sample: dict) -> str:
    model, label_encoder, feature_columns = load_artifacts()

    row = pd.DataFrame([sample])
    row = pd.get_dummies(row, drop_first=True)
    row = row.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(row)[0]
    return label_encoder.inverse_transform([prediction])[0]


def main() -> None:
    sample = {
        "Gender": "Male",
        "Age": 25.0,
        "Height": 1.75,
        "Weight": 80.0,
        "family_history_with_overweight": "yes",
        "FAVC": "yes",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "no",
        "CH2O": 2.0,
        "SCC": "no",
        "FAF": 1.0,
        "TUE": 1.0,
        "CALC": "Sometimes",
        "MTRANS": "Public_Transportation",
    }

    result = predict_one(sample)
    print(f"Предсказанный класс: {result}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка при предсказании: {exc}", file=sys.stderr)
        sys.exit(1)