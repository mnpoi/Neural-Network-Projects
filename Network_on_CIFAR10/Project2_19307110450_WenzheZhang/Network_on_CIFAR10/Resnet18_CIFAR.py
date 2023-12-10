import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import time
import re
#   加载gpu 如果没有gpu则使用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is %s' % device)


batch_size = 128
# 准备数据
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
        , transforms.RandomCrop(32, padding=4)  # 先四周填充0，图像随机裁剪成32*32
        , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
        , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# 准备数据 这里将训练集和验证集写到了一个list里 否则后面的训练与验证阶段重复代码太多
image_datasets = {
    x: torchvision.datasets.CIFAR10('./cifar10', train=True if x == 'train' else False,
               transform=data_transforms[x], download=True) for x in ['train', 'valid']}

dataloaders: dict = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False
    ) for x in ['train', 'valid']
}

# 基础块
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, LeakyReLU


class BasicBlock(nn.Module):

    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_features / self.in_features == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")

        self.conv1 = Conv2d(in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.leakyrelu = LeakyReLU(inplace=True)
        # self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(_features, _features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # 下采样
        self.downsample = None if self.in_features == self.out_features else nn.Sequential(
            Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        # out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 输入输出的特征数不同时使用下采样层
        if self.in_features != self.out_features:
            identity = self.downsample(x)

        # 残差求和
        out += identity
        out = self.leakyrelu(out)
        # out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_class=10) -> None:
        super().__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.leakyrelu = LeakyReLU(inplace=True)
        # self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = Linear(in_features=512, out_features=num_class, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        # x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # <---- 输出为{Tensor:(64,512,1,1)}
        x = torch.flatten(x, 1)  # <----------------压平 输出为{Tensor:(64,512)}
        x = self.fc(x)  # <------------ 输出为{Tensor:(64,10)}
        return x


model = ResNet18(num_class=10).to(device)
loss_fn =nn.CrossEntropyLoss()#交叉熵损失函数
total_epochs = 120
#print(model)

print('start training ... ')
#   存储训练过程的loss数值
train_loss = np.zeros(total_epochs)
test_loss = np.zeros(total_epochs)
#   存储训练过程的acc数值
train_acc = np.zeros(total_epochs)
test_acc = np.zeros(total_epochs)
# 用于记录损失值未发生变化batch数
counter = 0
# 记录当前最小损失值
valid_loss_min = np.Inf
model.train()
LR = 0.1  # 学习率

# 训练模型所需参数
# 用于记录损失值未发生变化batch数
counter = 0
# 记录训练次数
total_step = {
    'train': 0, 'valid': 0
}
# 记录开始时间
since = time.time()
# 记录当前最小损失值
valid_loss_min = np.Inf
# 保存模型文件的尾标
save_num = 0
# 保存最优正确率
best_acc = 0

for epoch in range(total_epochs):
    # 动态调整学习率
    if counter / 10 == 1:
        counter = 0
        LR = LR * 0.5


    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    print('Epoch {}/{}'.format(epoch + 1, total_epochs))
    print('-' * 10)
    print()
    # 训练和验证 每一轮都是先训练train 再验证valid
    for phase in ['train', 'valid']:
        # 调整模型状态
        if phase == 'train':
            model.train()  # 训练
        else:
            model.eval()  # 验证

        # 记录损失值
        running_loss = 0.0
        # 记录正确个数
        running_corrects = 0


        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                for param in model.parameters():
                    l2_reg = torch.norm(param, 2)
                loss += 0.0001 * l2_reg


                _, preds = torch.max(outputs, 1)  # 前向传播

                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()  # 反向传播
                    optimizer.step()  # 优化权重


            # 计算损失值
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum()  # 计算预测正确总个数
            total_step[phase] += 1

        # 一轮训练完后计算损失率和正确率
        epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
        epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率

        if phase == 'train':
            train_acc[epoch] = epoch_acc
            train_loss[epoch] = epoch_loss
        else:
            test_acc[epoch] = epoch_acc
            test_loss[epoch] = epoch_loss
        time_elapsed = time.time() - since
        print()
        print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format(phase, epoch_loss, counter, epoch_acc))

        if phase == 'valid':
            # 得到最好那次的模型
            if epoch_loss < valid_loss_min:

                best_acc = epoch_acc

                valid_loss_min = epoch_loss
                counter = 0
            else:
                counter += 1

    print('当前学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))

    if phase == 'valid' and epoch_acc >= 0.95:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, './model/model_checkpoint_' + str(epoch) + '.tar'
        )
        print("已保存最优模型，准确率:\033[1;31m {:.2f}%\033[0m，文件名：{}".format(epoch_acc * 100,
                                                                                 'model_checkpoint_' + str(
                                                                                     epoch) + '.tar'))

# 训练结束
time_elapsed = time.time() - since
print()
print('任务完成！')
print('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('最高验证集准确率: {:4f}'.format(best_acc))

# 画图
plt.figure(figsize=(16,8))
plt.plot(train_acc, 'o-', label='train_acc')
plt.plot(test_acc, 'o-', label='test_acc')
plt.legend()
plt.savefig('train_acc.jpg')
plt.show()

plt.figure(figsize=(16,8))
plt.plot(train_loss, 'o-', label='train_loss')
plt.plot(test_loss, 'o-', label='test_loss')
plt.legend()
plt.savefig('test_loss.jpg')
plt.show()
