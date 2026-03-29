FROM python:3.11-slim

# Install system libs required by gradio + matplotlib on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Verify gradio installed — build fails loudly here if not
RUN python -c "import gradio; print('gradio', gradio.__version__, 'OK')"

# Create HF-required non-root user
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    MPLBACKEND=Agg

WORKDIR $HOME/app

COPY --chown=user:user . $HOME/app

EXPOSE 7860

CMD ["python", "app.py"]
