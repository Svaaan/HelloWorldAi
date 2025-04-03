# Use official Python 3.11 slim base for smaller image size
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies first (if any)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (without caching to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your app
COPY . .

# Set environment variables
ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "src/app.py"]
