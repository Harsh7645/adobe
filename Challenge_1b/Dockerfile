FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Default command
CMD ["python", "src/main.py"]
