# ==== Base image ====
FROM python:3.11-slim

# No .pyc files, instant logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# ---- System deps (for things like OpenCV, numpy, etc.) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY . .

# Port uvicorn will listen on
EXPOSE 8000

# ---- Start FastAPI with Uvicorn ----
# If your entry is different, change "app.main:app" accordingly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
