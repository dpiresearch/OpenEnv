# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install numpy for pattern generation
RUN pip install --no-cache-dir numpy==1.24.0

# Copy core environment server code
COPY src/core/ /app/src/core/

# Copy stock patterns environment code
COPY src/envs/stock_patterns_env/ /app/src/envs/stock_patterns_env/

# Set working directory
WORKDIR /app/src

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "envs.stock_patterns_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

