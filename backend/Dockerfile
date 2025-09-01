# backend/Dockerfile

FROM python:3.9-slim

# Cài đặt các công cụ biên dịch cần thiết (gcc)
RUN apt-get update && apt-get install -y gcc build-essential

WORKDIR /app
COPY requirements.txt .

# Bây giờ lệnh này sẽ chạy thành công vì đã có gcc
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]