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

# Expose port
EXPOSE 8000

# Create startup script to handle bootstrap fallback
RUN echo '#!/bin/bash\n\
echo "Starting Twitter Bookmark Manager..."\n\
if [ -f "/app/bootstrap.py" ]; then\n\
  echo "Found bootstrap.py - attempting to run main application with bootstrap fallback..."\n\
  python -m gunicorn --bind 0.0.0.0:8000 --timeout 300 --workers 1 --threads 3 \\\n\
         --worker-class gthread --max-requests 1000 --max-requests-jitter 100 \\\n\
         bootstrap:app\n\
else\n\
  echo "No bootstrap.py found - running normal wsgi application..."\n\
  python -m gunicorn --bind 0.0.0.0:8000 --timeout 300 --workers 1 --threads 3 \\\n\
         --worker-class gthread --max-requests 1000 --max-requests-jitter 100 \\\n\
         wsgi_final:application\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 