FROM python:3.11-slim

# Copy uv from its official image for blazing-fast installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up a new user 'user' with UID 1000 (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements file first to maximize Docker layer caching
COPY --chown=user requirements.txt .

# Use uv to install packages (~10x-100x faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt || true

# Copy the rest of the application with proper ownership
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["python", "app.py"]
