import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# Load a 32x32 grayscale PNG and scale to [0, 1]
img = Image.open("label7.png").convert("L")
img = ImageOps.invert(img)

x = np.array(img, dtype=np.float32) / 255.0
x = x.reshape(1, 1, 32, 32)  # (N, C, H, W)

# Load the ONNX model and run inference
sess = ort.InferenceSession("my_lenet.onnx", providers=["CPUExecutionProvider"])
logits = sess.run(["logits"], {"input": x})

# Print the class with the largest logit
print("Predicted:", int(logits.argmax()))
