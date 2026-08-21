FROM python:3.10-slim

# Prevent Python from writing .pyc files
# and make logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# Install Python dependencies first.
# This allows Docker to cache this layer
# when only application code changes.
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir \
    -r /app/backend/requirements.txt

# Copy the complete backend.
COPY backend /app/backend

# Work from the backend directory so
# "app.main:app" resolves correctly.
WORKDIR /app/backend

# FastAPI
EXPOSE 8000

# Start the API server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
