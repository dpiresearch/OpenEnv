# Mastermind Environment Docker Image
# This Dockerfile creates a containerized version of the Mastermind environment
# that can be deployed and accessed via HTTP.

# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install any additional dependencies (none needed for Mastermind)
# RUN pip install --no-cache-dir <any-dependencies>

# Copy core framework code
COPY src/core/ /app/src/core/

# Copy Mastermind environment code
COPY src/envs/mastermind_env/ /app/src/envs/mastermind_env/

# Set working directory
WORKDIR /app/src

# Expose port for the HTTP server
EXPOSE 8000

# Health check to ensure server is running
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.mastermind_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

