# 将输出层的参数初始化，进行训练
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torchvision.transforms as transforms
import torch.utils.data as data
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import ImageFolder

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')

# 加载预训练模型
weights = ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 200)  # 修改输出层为200个类别

# 对新添加的最后一层进行初始化
nn.init.xavier_normal_(model.fc.weight)


# 定义数据变换（无数据增强）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='/mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011/images', transform=transform)

train_idx = []
test_idx = []
with open('/mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011/train_test_split.txt', 'r') as file:
    for line in file:
        index, is_train = map(int, line.strip().split())
        if is_train == 1:
            train_idx.append(index - 1)
        else:
            test_idx.append(index - 1)

train_dataset = data.Subset(dataset, train_idx)
test_dataset = data.Subset(dataset, test_idx)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])

class LitModel(LightningModule):
    def __init__(self, learning_rate, optimizer_name):
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("loss/val", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        return optimizer

learning_rate = 0.01  # 设置固定学习率为0.01
batch_size = 32
num_epochs = 20
optimizer_name = "SGD"
weight_decay = 0.0001

model = LitModel(learning_rate, optimizer_name)
logger = TensorBoardLogger("/mnt/ly/models/wy/results", name="train_initialize")

checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    mode='max',
    save_top_k=1,
    dirpath='/mnt/ly/models/wy/tune/checkpoints',
    filename='best-checkpoint'
)

trainer = Trainer(
    max_epochs=num_epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None,
    callbacks=[checkpoint_callback],
    logger=logger
)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

trainer.fit(model, train_loader, val_loader)
'''
加载测试集
class BestLitModel(LightningModule):
    def __init__(self, config):
        super(BestLitModel, self).__init__()
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 200)
        self.criterion = nn.CrossEntropyLoss()
        self.config = config

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": acc}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning rate"], momentum=0.9)
        return optimizer

best_model = BestLitModel.load_from_checkpoint(checkpoint_callback.best_model_path, config={
    'learning rate': learning_rate,
    'optim': optimizer_name
})

test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
trainer = Trainer(logger=logger)
test_results = trainer.test(best_model, test_loader)

print(f"Test Results - Loss: {test_results[0]['test_loss']}, Accuracy: {test_results[0]['test_accuracy']}")
'''
# 启动 TensorBoard
print(f"启动 TensorBoard: tensorboard --logdir=/mnt/ly/models/wy/results")
