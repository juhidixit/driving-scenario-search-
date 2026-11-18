FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ml]"
COPY configs ./configs

# Mount artifacts/, models/ and (optionally, for thumbnails) data/nuscenes/ at runtime:
#   docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts -v $PWD/models:/app/models scenario-search
ENV SCENARIO_ROOT=/app SCENARIO_ARTIFACTS=/app/artifacts SCENARIO_MODELS=/app/models NUSCENES_ROOT=/app/data/nuscenes
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "scenario_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
