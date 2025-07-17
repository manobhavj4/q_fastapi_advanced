# ======================
# Base Python Image
# ======================
FROM python:3.11-slim

# ======================
# Set Working Directory
# ======================
WORKDIR /app

# ======================
# Install system dependencies
# ======================
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ======================
# Copy requirements and install
# ======================
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ======================
# Copy source code
# ======================
COPY . .

# ======================
# Expose ports
# ======================
EXPOSE 8000 8501 8888

# ======================
# Default command
# ======================
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
