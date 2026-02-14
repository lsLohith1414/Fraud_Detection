FROM python:3.11-slim

WORKDIR /app

# Install system deps (lightgbm needs these sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only required folders
COPY app/ app/
COPY src/ src/
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
