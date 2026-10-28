FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ENCORE_CONFIG=configs/tiny.yaml PORT=8000

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY pyproject.toml README.md ./
COPY encore ./encore
RUN pip install --no-cache-dir .
COPY configs ./configs

EXPOSE 8000
CMD ["sh", "-c", "uvicorn --factory encore.serve.app:app_from_env --host 0.0.0.0 --port ${PORT}"]
