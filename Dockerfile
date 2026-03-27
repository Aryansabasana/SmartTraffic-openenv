FROM python:3.11-slim

WORKDIR /app

# Install Gradio and environment requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy all files
COPY . .

# Expose the standard Hugging Face Spaces port
EXPOSE 7860

# Run the Gradio Web application interface
CMD ["python", "app.py"]
