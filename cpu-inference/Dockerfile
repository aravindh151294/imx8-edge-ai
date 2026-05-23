FROM arm64v8/python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy<2" \
    tflite-runtime \
    pillow

COPY mobilenet_v1_1.0_224_quant.tflite .
COPY labels.txt .
COPY run_inference.py .
COPY test_image.jpg .

CMD ["python3", "run_inference.py", "test_image.jpg"]
