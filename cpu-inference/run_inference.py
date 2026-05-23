import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time
import sys

MODEL_PATH  = "mobilenet_v1_1.0_224_quant.tflite"
LABELS_PATH = "labels.txt"
IMAGE_PATH  = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(IMAGE_PATH).resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

start = time.perf_counter()
for _ in range(20):
    interpreter.invoke()
end = time.perf_counter()

avg_ms = ((end - start) / 20) * 1000
output_data = interpreter.get_tensor(output_details[0]['index'])
top_indices = np.argsort(output_data[0])[::-1][:5]

print(f"\n=== iMX8MP CPU Benchmark (4 threads) ===")
print(f"Average latency:  {avg_ms:.2f} ms")
print(f"FPS capability:   {1000/avg_ms:.1f}")
print(f"\nTop 5 predictions:")
for i, idx in enumerate(top_indices):
    confidence = output_data[0][idx] / 255.0 * 100
    print(f"  {i+1}. {labels[idx]:<40} {confidence:.1f}%")
