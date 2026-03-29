FROM python:3.11-slim

# Install system packages as root
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Verify gradio is installed (build will fail here if it's not — no silent failures)
RUN python -c "import gradio; print('gradio', gradio.__version__, 'OK')"

# Create HF-required non-root user
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app

WORKDIR $HOME/app

# Copy ALL project files with correct ownership
COPY --chown=user:user . $HOME/app

EXPOSE 7860

CMD ["python", "app.py"]
