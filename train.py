import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np

# load mnist
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# train
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss() #loss function

for epoch in range(3):
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.view(-1, 784) #flatten the image 
        outputs = model(images) #model output 
        loss = loss_fn(outputs, labels) #loss calc
        #backprop -----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #---------------
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"epoch {epoch+1} accuracy: {correct/total*100:.1f}%")

# save weights and biases
model[0].weight.detach().numpy().T.tofile("w1.bin")
model[2].weight.detach().numpy().T.tofile("w2.bin")
model[0].bias.detach().numpy().tofile("b1.bin")
model[2].bias.detach().numpy().tofile("b2.bin")

# save one test image to run inference on
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
image, label = test_data[0]
image.view(-1).numpy().tofile("image.bin")
print(f"saved test image, actual digit is: {label}")

print("done")