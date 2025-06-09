# -------- Stage 1: Build dependencies --------
FROM python:3.10-slim AS build

WORKDIR /app

# Install build tools (gcc) only here
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtualenv
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# -------- Stage 2: Final runtime image --------
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from build stage
COPY --from=build /install /usr/local

# Copy the MLflow model folder
COPY artifacts/model /app/artifacts/model

# Copy only necessary code and assets
COPY scripts/inference.py ./scripts/inference.py
COPY data/processed/test_set.csv ./test_set.csv
COPY src/housinglib ./src/housinglib

ENV PYTHONPATH=/app/src
ENV MODEL_URI=/app/artifacts/model
CMD ["python", "-u", "scripts/inference.py", "--no_console_log=False"]