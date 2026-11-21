FROM python:3.10-slim

WORKDIR /app

COPY artifacts/ /app/artifacts/
COPY app/app.py /app/app.py

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    scikit-learn==1.5.0 \
    pandas==2.2.2 \
    joblib==1.4.2 \
    numpy==1.26.4

ENV ARTIFACTS_DIR=/app/artifacts

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]