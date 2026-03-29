FROM python:3.11-slim

# Copy requirements file first to maximize Docker layer caching
COPY requirements.txt .

# Install packages using pip (reliable on Hugging Face Spaces)
RUN pip install --no-cache-dir -r requirements.txt

# Set up a new user 'user' with UID 1000 (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the rest of the application with proper ownership
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["python", "app.py"]
