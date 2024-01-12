import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np

# 定义超参数
input_size = 28  # 图像尺寸28*28
num_classes = 10  # 标签的种类数
num_epochs = 3  # 训练的总循环周期
BATCH_SIZE = 1024  # 设置批处理大小


# 数据处理成 tensor
def get_dataloader(train=True):
    transform_fn = Compose([
        ToTensor(),
        Normalize(
            mean=(0.1307,), std=(0.3081,)
        )
    ])

    # 导入MNIST训练数据集
    dataset = MNIST(root='./data', train=train, transform=transform_fn)
    # 构建batch数据
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader


# 构建训练数据
train_loader = get_dataloader()
# 构建测试数据
test_loader = get_dataloader(train=False)


# 构建CNN模块
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一子层卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入特征图个数1
                out_channels=16,  # 得到的特征图个数16
                kernel_size=5,  # 卷积核的大小
                stride=1,  # 步长
                padding=2,  # 填充
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2)
            # 最大池化
        )
        # 第二子层卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 输入特征图个数16
                out_channels=32,  # 输出特征图个数32
                kernel_size=5,  # 卷积核的大小
                stride=1,  # 步长
                padding=2,  # 填充
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2)  # 输出为 7, 7, 32
        )
        self.out = nn.Linear(32 * 7 * 7, 10)    #输出类别
        # pass
    # 前向传播
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0), -1)     # 拉直操作
        output=self.out(x)
        return output
    # 评估函数
    @staticmethod
    def accuracy(predictions, labels):
        pred = torch.max(predictions.data, 1)[1]
        rights = pred.eq(labels.data.view_as(pred)).sum()
        return rights, len(labels)

if torch.cuda.is_available()==False:
    print("CUDA不可用")
else:
    # 实例化
    net = CNN()
    if torch.cuda.is_available():
        net = net.cuda()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 普通的随机梯度下降算法

    # 开始训练
    for epoch in range(num_epochs):
        # 用于保存当前epoch的结果
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):   # 针对容器里的每一批进行循环
            net.train()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()  # Move data to GPU
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = net.accuracy(output, target)
            train_rights.append(right)
        # 每隔100次，在验证机上检验效果
            if batch_idx%100 ==0:
                net.eval()
                val_rights = []

                for (data, target) in test_loader:
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()  # Move data to GPU
                    output = net(data)
                    right = net.accuracy(output, target)
                    val_rights.append(right)

                # 计算准确率
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失:{:.6f}\t训练准确率:{:.2f}%\t训练正确率:{:.2f}%'.format(
                    epoch, batch_idx*BATCH_SIZE, len(train_loader.dataset),
                    100.*batch_idx / len(train_loader),
                    loss.data,
                    100.*train_r[0].cpu().numpy() / train_r[1],
                    100.*val_r[0].cpu().numpy() / val_r[1]))



