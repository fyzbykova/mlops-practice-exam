# MLOps Practice Exam

## Задание
Практическая часть теста по MLOps (вопросы 46–50).

## Датасет
**Obesity Prediction Dataset** (Kaggle).
- 2111 строк, 17 колонок
- Целевая переменная: `NObeyesdad` (7 классов ожирения)
- Источник: https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset

## Модель
`RandomForestClassifier(n_estimators=100, random_state=42)`.
Accuracy на тестовой выборке: **0.9385**.

## Структура
- `notebooks/training.ipynb` — обучение модели и логирование в MLflow
- `src/` — скрипты для воспроизведения
- `artifacts/` — модель, энкодер, список признаков, датасет
- `app/` — FastAPI-сервис и Dockerfile
- `docs/answers.md` — ответы на вопросы 49 и 50

## Запуск

### 1. Обучение
```bash
pip install -r requirements.txt
python src/download_data.py
python src/train.py