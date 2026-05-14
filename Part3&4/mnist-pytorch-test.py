import torch
import torch.nn as nn
import torchvision

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    device = get_device()
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("best_mnist_model.pth"))
    test(model, device, test_loader)