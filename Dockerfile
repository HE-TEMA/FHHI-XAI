# Optimized Dockerfile for Python application
# Uses multi-stage build and improved layer caching

# Build stage for installing dependencies
FROM python:3.10-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    # Therse two are for h5py
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy and install requirements first (better layer caching)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Final stage
FROM python:3.10-slim-bookworm

# Container settings only; application defaults live in the Python code.
# Override explanation settings at runtime with docker run -e when needed.
ENV PORT=8080 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PERSON_VEHICLE_CONFIG=/app/output/production/person_vehicle_stem_p5_20261012/k4.json

WORKDIR /app

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    redis-server \
    supervisor \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/log/supervisor /var/lib/redis \
    && chown -R redis:redis /var/lib/redis \
    && rm -f /var/lib/redis/dump.rdb

# Copy wheels from builder stage and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Configure supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy application code (done last to maximize caching)
COPY . .

# Expose port
EXPOSE ${PORT}

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
