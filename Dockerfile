FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    PORT=9696 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies exactly as locked in uv.lock (no dev tools)
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 uv
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# Copy the code and data, then train the model inside the image so it always matches the installed scikit-learn
COPY common.py train.py predict.py ./
COPY data ./data
RUN python train.py

EXPOSE 9696

# $PORT lets managed platforms (e.g. Google Cloud Run) choose the port; defaults to 9696
CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]
