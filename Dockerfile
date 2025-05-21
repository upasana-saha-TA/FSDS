# Use an official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install only system dependencies required by your libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy  requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy only what’s needed
COPY scripts/inference.py ./scripts/inference.py
COPY artifacts/model_pickle ./model_pickle
COPY data/processed/test_set.csv ./test_set.csv
COPY src/housinglib ./src/housinglib

# Set the Python path so it can find housinglib
ENV PYTHONPATH=/app/src

# Run the inference script (can also use CMD to make it overridable)
CMD ["python", "scripts/inference.py"]

