FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CARTIGSFM_WEB_HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY pyproject.toml setup.py README.md LICENSE ./
COPY cartigsfm ./cartigsfm
COPY cartigsfm_web ./cartigsfm_web

RUN pip install --no-cache-dir -e ".[web]"

EXPOSE 8000

CMD ["python", "-m", "cartigsfm_web.server"]
