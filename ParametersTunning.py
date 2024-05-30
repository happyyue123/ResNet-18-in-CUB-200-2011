import torch

# 设置矩阵乘法精度，以利用 Tensor Cores 性能
torch.set_float32_matmul_precision('medium')  # 或者 'medium'，根据需要选择

import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import os
import numpy as np
import random
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

from pytorch_lightning.loggers import TensorBoardLogger


# 使用ResNet_18作为训练模型，并加载预训练参数
weights = ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 200)  # 修改输出层为200个类别

# 对新添加的最后一层进行初始化
nn.init.xavier_normal_(model.fc.weight)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ImageFolder(root='/mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011/images', transform=transform)

# 读取训练和测试分割信息
train_idx, test_idx = [], []
with open('/mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011/train_test_split.txt', 'r') as file:
    for line in file:
        index, is_train = map(int, line.strip().split())
        if is_train == 1:
            train_idx.append(index - 1)  # 减1因为文件序号通常从1开始，而Python列表索引从0开始
        else:
            test_idx.append(index - 1)

# 创建数据子集
train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# 将train_dataset划分一部分为val_set
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.config["optim"] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning rate"], momentum=0.9, weight_decay=self.config['weight decay'])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning rate"], weight_decay=self.config['weight decay'])
        return optimizer

logger = TensorBoardLogger(save_dir='/mnt/ly/models/wy/results/checkpoints', name="", version=".")

def train_mnist(config, checkpoint_dir='/mnt/ly/models/wy/results/checkpoints'):
    model = LitModel(config)
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[TuneReportCallback({'val_loss': 'val_loss', 'val_accuracy': 'val_accuracy'})],
        logger=logger
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch size'], shuffle=False, num_workers=4)
    trainer.fit(model, train_loader, val_loader)

def parameters_train(config):
    # 去掉scheduler，确保所有实验都完整运行
    result = tune.run(
        train_mnist,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=10,
        local_dir='/mnt/ly/models/wy/tune',
        name="mid_term_parameters_results"
    )

    best_trial = result.get_best_trial("val_accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_accuracy"]))

if __name__ == '__main__':
    torch.manual_seed(42)  # 设置随机种子
    config = {
        'learning rate': tune.choice([0.1, 0.01, 0.001]),
        'batch size': tune.choice([16, 32, 64]),
        'optim': tune.choice(['SGD', 'Adam']),
        'weight decay': tune.loguniform(1e-5, 1e-1)  # 纠正拼写错误，使用 loguniform 而不是 lograndint
    }

    parameters_train(config)
