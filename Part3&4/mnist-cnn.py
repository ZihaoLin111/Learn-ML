import torch
import torch.nn as nn
import torchvision
import tqdm
import matplotlib.pyplot as plt
import datetime


class CNN_For_MNIST(nn.Module):
    def __init__(self):
        super(CNN_For_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()

    acc = correct / len(val_loader.dataset)
    acc_list.append(acc)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model_autoF2.pth")


def train_epoch_with_handwriteL2(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    l2_lambda,
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
        l2_weight = sum(torch.sum(p**2) for p in model.parameters())
        loss += l2_lambda * l2_weight / 2
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
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
    acc = correct / len(val_loader.dataset)
    acc_list.append(acc)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model_handwriteF2.pth")


def show_loss(loss):
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(f"mnist_cnn_loss_{timestamp}.png")
    plt.show()


def show_acc(acc):
    plt.plot(acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Testing Accuracy")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(f"mnist_cnn_accuracy_{timestamp}.png")
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
    model_1 = CNN_For_MNIST().to(device)

    # 第一种方法，使用Pytorch的weight_decay参数

    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.1, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loss_1 = []
    acc_1 = []
    best_acc_1 = 0

    for epoch in range(10):
        train_epoch(
            model_1,
            device,
            train_loader,
            val_loader,
            optimizer_1,
            criterion,
            loss_1,
            acc_1,
            best_acc_1,
        )

        print(
            f"Epoch: {epoch + 1}, Loss: {loss_1[-1]:.4f}, Accuracy: {acc_1[-1]:.4f}"
        )
    show_acc(acc_1)
    show_loss(loss_1)

    # 第二种方法，自定义L2正则化

    model_2 = CNN_For_MNIST().to(device)
    l2_lambda = 1e-4
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    loss_2 = []
    acc_2 = []
    best_acc_2 = 0

    for epoch in range(10):
        train_epoch_with_handwriteL2(
            model_2,
            device,
            train_loader,
            val_loader,
            optimizer_2,
            criterion,
            l2_lambda,
            loss_2,
            acc_2,
            best_acc_2,
        )

        print(
            f"Epoch: {epoch + 1}, Loss: {loss_2[-1]:.4f}, Accuracy: {acc_2[-1]:.4f}"
        )

    show_acc(acc_2)
    show_loss(loss_2)
