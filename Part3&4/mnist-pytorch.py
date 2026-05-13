import torch
import torch.nn as nn
import torchvision
import mlflow

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
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
    
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

def mlflow_autolog():
    # 自动将常用的训练参数/指标等记录到MLflow中
    mlflow.pytorch.autolog()

if __name__ == "__main__":
    train_dataset = torchvision.datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=torchvision.transforms.ToTensor())

    test_dataset = torchvision.datasets.MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False)
    device = get_device()
    model = NeuralNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(10):
        train(model, device, train_loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}/10 completed.")
    
