# Serving image: the SAME image runs locally (docker-compose) and on SageMaker (BYOC, port 8080, `serve` entrypoint).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    MODEL_DIR=/opt/ml/model PORT=8080

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY serving ./serving
COPY docker/serve /usr/local/bin/serve

RUN chmod +x /usr/local/bin/serve \
 && useradd --create-home --uid 1000 app \
 && mkdir -p /app/logs /opt/ml/model \
 && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/ping').status==200 else 1)"

# The model is NOT baked in: SageMaker mounts model.tar.gz at /opt/ml/model, compose mounts ./models.
ENTRYPOINT ["serve"]
