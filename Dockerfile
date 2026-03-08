# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Sistem bağımlılıkları (XGBoost derleme için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Build stage'den kurulmuş paketleri kopyala
COPY --from=builder /install /usr/local

# Uygulama dosyalarını kopyala
COPY app/    ./app/
COPY models/ ./models/

# .env production'da mount edilmeli, image'a gömme
# ENV değişkenleri docker run -e veya docker-compose ile verilir

EXPOSE 8000

# Non-root user (güvenlik)
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
