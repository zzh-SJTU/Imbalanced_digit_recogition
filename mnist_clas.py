import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.vision.datasets import MNIST
import os
import random
from paddle.vision.transforms import ToTensor
import paddle.nn.functional as F
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

def train(model, opt, train_loader, valid_loader,gamma_list,dic_acc,dic_loss):
    # 开启0号GPU训练
    for gamma in gamma_list:
        use_gpu = True
        paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
        print('start training ... ')
        model.train()
        for epoch in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                img = data[0]
                label = data[1] 
                if batch_id % 10 != 0:
                    mask = (label >= 5)
                else:
                    mask = (label >= 0)
                img = img[mask]
                if img.shape[0] == 0:
                    continue
                img = img.reshape((-1,1,28,28))
                
                label = label[mask]
                
                # 计算模型输出
                logits = model(img)
                # 计算损失函数
                loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(logits, label)
                
                with paddle.no_grad():
                    pt = paddle.exp(-loss)
                total_loss = (1-pt)**gamma*loss
                
                avg_loss = paddle.mean(total_loss)
                if batch_id % 2000 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
                avg_loss.backward()
                opt.step()
                opt.clear_grad()

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                img = data[0]
                label = data[1] 
                # 计算模型输出
                logits = model(img)
                pred = F.softmax(logits)
                # 计算损失函数
                loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(logits, label)
                acc = paddle.metric.accuracy(pred, label)
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
            #dic_acc[gamma].append(np.mean(accuracies))
            #dic_loss[gamma].append(np.mean(losses))
            #print(gamma)
            model.train()

model = LeNet(num_classes=10)
# 设置迭代轮数
EPOCH_NUM = 10
# 设置优化器为Momentum，学习率为0.001
opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
# 定义数据读取器
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=100, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=100)
# 启动训练过程
dic_acc = {}
dic_loss = {}
gamma_list = [0.2,0.5,0.8,1,1.5,2,2.5,3,4,5,10,15]
for gamma in gamma_list:
    dic_acc[gamma] = []
    dic_loss[gamma] = []
train(model, opt, train_loader, valid_loader,gamma_list,dic_acc,dic_loss)
