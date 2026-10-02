FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir uv \
    && uv sync --no-dev

COPY scripts/inference/ scripts/inference/

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

CMD ["uv", "run", "python", "scripts/inference/predict_model.py"]