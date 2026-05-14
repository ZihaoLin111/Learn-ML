import torch
import torch.nn as nn
import torchvision
import tqdm
import matplotlib.pyplot as plt
import datetime


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


def train_epoch(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    loss_list,
    acc_list,
    best_acc,
):
    model.train()
    train_loss = 0
    for data, label in tqdm.tqdm(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss / len(train_loader)
    loss_list.append(avg_loss)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in tqdm.tqdm(val_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    acc = correct / len(val_loader.dataset)
    acc_list.append(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_mnist_model.pth")


def show_loss(loss):
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(f"mnist_pytorch_loss_{timestamp}.png")
    plt.show()


def show_acc(acc):
    plt.plot(acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Testing Accuracy")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(f"mnist_pytorch_accuracy_{timestamp}.png")
    plt.show()


if __name__ == "__main__":
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    device = get_device()
    model = NeuralNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    loss = []
    acc = []
    best_acc = 0
    for epoch in range(10):
        train_epoch(
            model,
            device,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            loss_list=loss,
            acc_list=acc,
            best_acc=best_acc,
        )
        print(f"Epoch {epoch + 1}:")
        print(f"Train Loss: {loss[-1]:.4f}, Validation Accuracy: {acc[-1]:.4f}")
    show_loss(loss)
    show_acc(acc)
