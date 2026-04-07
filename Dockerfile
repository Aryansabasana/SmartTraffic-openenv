FROM python:3.10-slim

# Build stability optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install isolated dependencies first to maximize Docker caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code after dependencies to prevent cache invalidation on code change
COPY . /app/

EXPOSE 7860

CMD ["python", "server/app.py"]
