FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project specifications and install dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel && \
    pip install .[dev,ml]

# Copy platform source tree
COPY . .

# Expose FastAPI REST, Dashboard & Prometheus port
EXPOSE 8000

# Default command starts Uvicorn ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
