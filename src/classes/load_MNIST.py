from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_MNIST(batch_size):
    """
    Load MNIST dataset
    A dataset of handwritten digits
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    traindataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    testdataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader