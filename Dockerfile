<<<<<<< HEAD
FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]
=======
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["gunicorn", "--chdir", "flask", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
>>>>>>> b2ed1211c4203beb27b1fa02ec0c7a16ee8bcb47
