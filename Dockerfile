# Dockerfile cho service_search
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_search.py .

# Create FAISS directory
RUN mkdir -p /app/faiss_data

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api_search:app", "--host", "0.0.0.0", "--port", "8000"]