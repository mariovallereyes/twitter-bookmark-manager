FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV DISABLE_VECTOR_STORE=true
ENV FLASK_DEBUG=false

# Expose port
EXPOSE 8000

# Run the bare minimum application
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "30", "--workers", "1", "--threads", "1", "--log-level", "debug", "wsgi_final:application"] 