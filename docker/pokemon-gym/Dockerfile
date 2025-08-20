FROM python:3.10-slim

WORKDIR /app

# Install minimal dependencies for fast startup
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn

# Copy mock service
COPY mock_service.py .

# Expose port
EXPOSE 8080

# Fast startup with single worker for minimal overhead
CMD ["uvicorn", "mock_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
