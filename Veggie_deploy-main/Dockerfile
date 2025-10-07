# ใช้ Python 3.12 slim เป็น base
FROM python:3.12-slim

# ติดตั้ง lib พื้นฐานและ OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ตั้ง working directory
WORKDIR /app

# คัดลอก requirements.txt
COPY requirements.txt .

# อัพเกรด pip และติดตั้ง dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมด
COPY . .

# Expose port 5000
EXPOSE 5000

# รันด้วย Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "3"]
