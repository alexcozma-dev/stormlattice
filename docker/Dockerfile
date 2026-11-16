# The API only. nginx on the VM serves web/ itself and proxies /api/ and /health here.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY stormlattice/ stormlattice/
COPY config/ config/

# web/ is not copied: compose mounts the checkout's web/ at /app/web, read-only.
# Run as an unprivileged user; /data is the only place it may write.
RUN useradd --system --uid 10001 storm && mkdir /data && chown storm /data
USER storm

ENV STORM_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "stormlattice.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
