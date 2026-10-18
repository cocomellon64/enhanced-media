FROM python:3.11-slim

WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required directories
RUN mkdir -p checkpoints feedback_data training_logs

ENV CHECKPOINT_DIR=/app/checkpoints
ENV FEEDBACK_DIR=/app/feedback_data
ENV ALLOW_FEEDBACK=true

EXPOSE 8000

CMD ["uvicorn", "enhancer.api.server:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
