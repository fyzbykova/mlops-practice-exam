"""Скачивание датасета Obesity Prediction с Kaggle и сохранение в artifacts/."""
import os
import sys

import kagglehub
import pandas as pd

ARTIFACTS_DIR = "artifacts"
DATASET_SLUG = "adeniranstephen/obesity-prediction-dataset"
CSV_NAME = "ObesityDataSet_raw_and_data_sinthetic.csv"


def download_dataset() -> pd.DataFrame:
    """Скачивает датасет через kagglehub и возвращает DataFrame."""
    path = kagglehub.dataset_download(DATASET_SLUG)
    csv_path = os.path.join(path, CSV_NAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV не найден по пути: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = download_dataset()

    out_path = os.path.join(ARTIFACTS_DIR, "obesity_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"Датасет сохранён: {out_path}")
    print(f"Размер: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка при скачивании датасета: {exc}", file=sys.stderr)
        sys.exit(1)