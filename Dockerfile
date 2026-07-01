FROM python:3.12-slim

WORKDIR /app

# Install core deps (BM25 retrieval; no torch).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the normalized catalog at image build time so startup is fast.
RUN python -m scripts.build_catalog

ENV PORT=8000
EXPOSE 8000

# Respect the platform-provided $PORT (Render/Railway set it); default 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
