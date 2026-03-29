FROM python:3.11-slim

# 1. OPTIMIZED SYSTEM LAYER: Consolidate apt-get and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 2. OPTIMIZED DEPENDENCY LAYER: Fast-track scientific resolution
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --prefer-binary -r /tmp/requirements.txt

# Verify installation immediately to fail fast
RUN python -c "import gradio; print('gradio', gradio.__version__, 'ready')"

# 3. PRODUCTION USER LAYER: HF-defined non-root best practices
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    MPLBACKEND=Agg

WORKDIR $HOME/app

# 4. APPLICATION LAYER: Final copy and perms
COPY --chown=user:user . $HOME/app

EXPOSE 7860
CMD ["python", "app.py"]
