# Multi-stage build for production deployment
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    wget \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies.
# libglib2.0-0/libgomp1 look like build-only packages but are NOT: the
# opencv-python-headless wheel still dynamically links libgthread-2.0.so.0 and
# libgomp.so.1. They were installed in the builder stage only, which never
# imports cv2 -- so `import cv2` raised ImportError on the first request path
# at startup and the container exited before the health check could pass.
# tesseract-ocr provides the binary pytesseract shells out to; the -dev headers
# are build-time only and are deliberately not carried into this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY templates/ ./templates/
# Exclude data in production to keep image small
# COPY data/ ./data/
COPY simple_server.py .

# Create necessary directories
RUN mkdir -p logs models data/uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TORCH_DEVICE=cpu

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Run the application — honor $PORT (Render/Heroku set it), default 8000.
# Worker count is env-driven and defaults to 1: each worker loads its own copy
# of cv2 + numpy + PyMuPDF (~300MB RSS), so a hardcoded 2 exceeded the 512MB
# free-instance limit and the service was OOM-killed in a restart loop. Raise
# WEB_CONCURRENCY on a paid instance with more memory.
CMD ["sh", "-c", "uvicorn simple_server:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]