FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stormlattice/ stormlattice/
COPY config/ config/
COPY web/ web/

ENV STORM_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "stormlattice.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
