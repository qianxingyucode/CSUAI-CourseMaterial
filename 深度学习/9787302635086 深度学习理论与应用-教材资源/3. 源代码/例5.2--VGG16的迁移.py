import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = models.vgg16(pretrained=True).to(device)
......  # 【此处为本例核心代码，已省略，完整代码见教材P134；建议读者手工输入核心代码并进行调试，这样方能领会其含义】


L = [conv2,conv3,conv5]  #对这些网络层上的参数进行冻结
for layer in L:
    for param in layer.parameters():
        param.requires_grad = False


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        # 全连接层
        self.fc1 = nn.Linear(512*6*6, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):  #torch.Size([16, 1, 224, 224])
        o = x
        o = self.conv1(o)   # torch.Size([16, 3, 222, 222])
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)  #torch.Size([16, 3, 111, 111])

        o = self.conv2(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = self.conv3(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = self.conv4(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = self.conv5(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = o.reshape(x.size(0),-1)

        o = self.fc1(o)  #全连接层
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)
        o = self.fc2(o)  #全连接层
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)
        o = self.fc3(o)  #全连接层
        return o

net = Net().to(device)
x = torch.randn(16, 1, 224, 224).to(device) #随机产生测试数据
y = net(x)  #调用网络模型

param_sum = 0               #统计参数总数
trainable_param_sum = 0     #统计可训练的参数总数
for param in net.parameters():
     n = 1
     for j in range(len(param.shape)):#统计当前层的参数个数
         n = n*param.size(j)
     param_sum += n
     if param.requires_grad:
         trainable_param_sum += n
print('该模型的参数总数为：{:.0f}，其中可训练的参数总数为：\
      {:.0f}，占的百分比为：{:.2f}%'.\
      format(param_sum,trainable_param_sum,\
      100.*trainable_param_sum/param_sum))

print('输入和输出的形状分别为：', x.shape,y.shape)





