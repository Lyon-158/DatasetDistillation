# %matplotlib inline
import random
import torch
from d2l import torch as d2l


# 人工生成数据集
def sysnthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 随机噪声
    return X, y.reshape((-1, 1))  # 将X，y作为列向量返回


# 真实值
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = sysnthetic_data(true_w, true_b, 1000)


# 每次读取一个小批量，接收批量大小、特征矩阵和标签向量，生成大小为batch_size的小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 读取下标
    indices = list(range(num_examples))  # 下标转换为列表
    random.shuffle(indices)  # 把下标随机打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        # params传入的是列表，包含(w,b)
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 1
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    # 对一个小批量
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()  # backward是对函数里所有required_grad=True的参数都求了梯度
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1},loss {float(train_l.mean()):f}')

# 输入f就代表花括号里表达式可以用表达式的值代替
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
