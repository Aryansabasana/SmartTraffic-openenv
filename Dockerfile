FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    MPLBACKEND=Agg

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

RUN python -c "import gradio; print('gradio', gradio.__version__, 'ready')"

COPY --chown=user:user . /home/user/app

EXPOSE 7860
# Use uvicorn directly if needed, but python app.py works with the current setup
CMD ["python", "app.py"]
