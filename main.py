import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import toml


# Final Validation Accuracy 88.83%
# The best working code at the moment without orthonormal U and V
# With config read function

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

traindataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
testdataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

trainloader = DataLoader(traindataset, batch_size=batchSize, shuffle=True)
testloader  = DataLoader(testdataset, batch_size=batchSize, shuffle=False)


criterion = nn.CrossEntropyLoss()
# Create an instance of NeuralNetwork and directly use the activation functions from the Activation class
NeuralNet = NeuralNetwork(config_layers, lr)

# Training loop
for i in range(num_epochs):
    for step, (images, labels) in enumerate(trainloader):
        out = NeuralNet(images)

        loss = criterion(out, labels)
        loss.backward()

        NeuralNet.update()

        # TODO: Train your weights and biases. Think about how you can use object oriented programming to do so.
        # The way you do this heavily impacts the extendability of your code. So take some time to think about the design of your program!

        if (step + 1) % 100 == 0:
            print(f'Epoch [{i+1}/{num_epochs}], Step [{step+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = NeuralNet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch [{i+1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}%')


