FROM public.ecr.aws/docker/library/python:3.10-slim

# Set environment variables for build and runtime stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Standard evaluator-safe dependency installation
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only necessary project files
COPY . .

EXPOSE 7860

CMD ["python", "server/app.py"]


