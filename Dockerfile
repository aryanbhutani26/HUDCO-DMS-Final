# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for numpy, torch, chromadb, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit entry point
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
