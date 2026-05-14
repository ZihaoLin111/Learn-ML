## 什么是MLflow？
> MLflow 是一个开源平台，专门为协助机器学习从业者和团队处理机器学习过程的复杂性而构建。MLflow 专注于机器学习项目的完整生命周期，确保每个阶段都可管理、可追溯且可重现。

## 日志记录

### 自动记录
``` python
# 自动将常用的训练参数/指标等记录到MLflow中
mlflow.pytorch.autolog()
```
MLflow的autolog支持Pytorch Lightning，对于常用的Pytorch Vanilla需要使用手动记录的方式。
**什么是Pytorch Lightning?** Pytorch Lightning是一个在Pytorch基础上构建的封装库，旨在简化深度学习训练过程。参考如下代码示例

``` python
import torch
import torch.nn as nn
import torchvision
import lightning as L
import mlflow

mlflow.pytorch.autolog()


class NeuralNet(L.LightningModule):
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

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nn.CrossEntropyLoss()(output, label)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nn.CrossEntropyLoss()(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)


if __name__ == "__main__":
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
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

    model = NeuralNet()
    trainer = L.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
```

### 手动记录

#### 记录参数
``` python
params = {
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "hidden_size": 128,
    "dropout": 0.3,
}
mlflow.log_params(params)               # 批量记录
mlflow.log_param("device", str(device)) # 单个记录
mlflow.log_param("model_architecture", "3-layer MLP with dropout")
```