import torch
import torch.nn as nn
import torch.quantization as tq
import time
import numpy as np
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x


# Load raw float32 binaries (same files your C++ code uses)
w1_np = np.fromfile("w1.bin", dtype=np.float32).reshape(784, 128)
w2_np = np.fromfile("w2.bin", dtype=np.float32).reshape(128, 10)
b1_np = np.fromfile("b1.bin", dtype=np.float32).reshape(128)
b2_np = np.fromfile("b2.bin", dtype=np.float32).reshape(10)

model_fp32 = MNISTNet()

# nn.Linear expects weight shape [out_features, in_features]
# your w1 is [in, out] -> transpose to [out, in]
model_fp32.fc1.weight.data = torch.from_numpy(w1_np.T).contiguous()
model_fp32.fc1.bias.data = torch.from_numpy(b1_np)
model_fp32.fc2.weight.data = torch.from_numpy(w2_np.T).contiguous()
model_fp32.fc2.bias.data = torch.from_numpy(b2_np)
model_fp32.eval()

# Sanity check against your C++ pipeline
test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
img, label = test_data[0]
with torch.no_grad():
    pred = model_fp32(img.view(1, -1)).argmax(dim=1).item()
print(f"Sanity check -- PyTorch prediction: {pred}, actual: {label}")

# Quantization setup
model_fp32.qconfig = tq.get_default_qconfig('fbgemm')  # use 'qnnpack' on ARM/Mac
model_prepared = tq.prepare(model_fp32)

with torch.no_grad():
    for i in range(100):
        img, _ = test_data[i]
        model_prepared(img.view(1, -1))

model_int8 = tq.convert(model_prepared)

# Accuracy check on held-out 200 images (indices 100-299)
correct_fp32 = 0
correct_int8 = 0
with torch.no_grad():
    for i in range(100, 300):
        img, label = test_data[i]
        img = img.view(1, -1)
        pred_fp32 = model_fp32(img).argmax(dim=1).item()
        pred_int8 = model_int8(img).argmax(dim=1).item()
        correct_fp32 += (pred_fp32 == label)
        correct_int8 += (pred_int8 == label)

print(f"FP32 accuracy: {correct_fp32}/200")
print(f"INT8 accuracy: {correct_int8}/200")

# Timing comparison
N = 200
imgs = [test_data[i][0].view(1, -1) for i in range(100, 300)]

with torch.no_grad():
    start = time.perf_counter()
    for img in imgs:
        model_fp32(img)
    fp32_time = time.perf_counter() - start

    start = time.perf_counter()
    for img in imgs:
        model_int8(img)
    int8_time = time.perf_counter() - start

print(f"FP32 time: {fp32_time*1000:.2f} ms ({fp32_time/N*1e6:.2f} us/image)")
print(f"INT8 time: {int8_time*1000:.2f} ms ({int8_time/N*1e6:.2f} us/image)")
print(f"Speedup: {fp32_time/int8_time:.2f}x")