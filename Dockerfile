FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install --no-cache-dir uv

COPY . /app
RUN uv pip install --system -e .

EXPOSE 8000
ENV HOST=0.0.0.0
ENV PORT=8000
ENV RERANKER=colbert

CMD ["python", "apps/api.py"]
