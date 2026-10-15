# Use Python 3.12-slim as base image
# Note: Local environment uses Python 3.14, but 3.14 Docker images may not be stable yet.
# Python 3.12 is stable, well-supported, and compatible with our dependencies.
# Python 3.11 was too old for many packages (contourpy, numpy, scipy require 3.12+).
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies FIRST
# This allows Docker to cache this layer. If requirements.txt doesn't change,
# Docker won't reinstall dependencies on subsequent builds, speeding up rebuilds.
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir prevents pip from caching downloaded packages, keeping image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary project files and directories
# We only copy what's needed to run the API, not development artifacts.
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the FastAPI application with uvicorn
# --host 0.0.0.0: Bind to all network interfaces inside the container.
#   Required because containers have their own network namespace.
#   If we used 127.0.0.1, it would only accept connections from within the container,
#   not from outside (e.g., from the host machine or other containers).
# --port 8000: The port the app will listen on inside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
