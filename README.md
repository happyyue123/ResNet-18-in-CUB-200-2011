# ResNet-18-in-CUB-200-2011
该项目使用CUB-200-2011数据集，使用ResNet模型预训练，最终准确率达到72%
训练好的模型权重：https://drive.google.com/file/d/1BFRt31_3H-BtqxhWupXihqIyhStA4_F1/view?usp=drive_link。
将训练好的模型权重放入自己的Google drive中，加载模型权重.ipynb，导入模型权重和数据集，可以查看最终训练效果。
## 预训练过程
剩下的3个py文件是我预训练模型的过程。parameterTunning.py文件通过选择不同的参数组合，在5epochs下短暂训练，查看表现较好的参数组合，放入learning rate change.py文件中，跑20-50epochs查看效果，并根据效果微调学习率，最终得到的超参数组合是：{learning rate：0.01，batch size:32, optim:SGD},并且不对输出层进行随机初始化，在测试集上的准确率是72%。weight initialize.py是完全不用Resnet-18预训练的模型参数训练的文件代码，跑出结果在测试集上的准确率只有22%。
