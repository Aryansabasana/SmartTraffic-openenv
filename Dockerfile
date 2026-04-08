FROM public.ecr.aws/docker/library/python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# ✅ Add this line — forces cache bust on every build
ARG CACHEBUST=1

COPY . .

EXPOSE 7860

CMD ["python", "server/app.py"]# force rebuild Wed, Apr  8, 2026  9:22:45 PM
