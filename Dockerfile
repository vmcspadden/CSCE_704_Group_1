# Static PE malware detector (Pipeline A: LSH pre-filter -> per-bucket RF).
# Build:  docker build -t pe-detector .
# Run:    docker run --rm -p 8080:8080 --memory=1g pe-detector
# Query:  curl -XPOST --data-binary @file.exe -H "Content-Type: application/octet-stream" http://127.0.0.1:8080/

FROM python:3.11-slim AS build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && rm -rf /opt/venv/lib/python3.11/site-packages/*/tests

FROM python:3.11-slim
COPY --from=build /opt/venv /opt/venv
RUN groupadd -r defender && useradd --no-log-init -r -g defender defender
WORKDIR /opt/detector
COPY src ./src
COPY defender ./defender
USER defender
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/detector" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    DF_PORT=8080 \
    DF_WORKERS=2 \
    DF_REQUEST_BUDGET=4.5
EXPOSE 8080
CMD ["python", "-m", "defender"]
