import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from efficientnet_pytorch import EfficientNet
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
from torchvision import datasets, transforms, models
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#---------------------------------
# ============= 以下开始加载数据并打包 =====================
path = r'./data/flower_photos'
#以下函数获取指定目录所有文件名（含路径）及其所属类的编号（每个目录一个类）
def getFileLabel(tmp_path):
    dirs = list(os.walk(tmp_path))[0][1]
    L = []
    for label, dir in enumerate(dirs):
        path2 = os.path.join(tmp_path,dir)
        files = list(os.walk(path2))
        for file in files[0][2]: #files[0][2]为path2目录下的所有文件
            fn = os.path.join(path2,file)
            if os.path.exists(fn):
                t = (fn,label)
                L.append(t)
    return L
file_labels = getFileLabel(path)
random.shuffle(file_labels)
random.shuffle(file_labels)  #打乱顺序
#按7：3划分训练集和测试集：
rate = 0.7
train_length = int(rate*len(file_labels))
train_file_labels = file_labels[:train_length]
test_file_labels = file_labels[train_length:]
#train_file_labels,test_file_labels  = train_file_labels[0:150],test_file_labels[0:150]  #暂时缩短
transform = transforms.Compose([
    transforms.Resize((224,224)),  #调整图像大小为(224,224)
    transforms.ToTensor(),  #转化张量
])
class FlowerDataSet(Dataset): #构建数据集类
    def __init__(self, data_file_label):   #
        self.data_file_label = data_file_label

    def __len__(self):     #需要重写该方法，返回数据集大小
        t = len(self.data_file_label)
        return t
    def __getitem__(self, idx):
        fn, label = self.data_file_label[idx][0], self.data_file_label[idx][1]
        img = Image.open(fn).convert('RGB')  # (600, 800, 3)
        img = transform(img)
        return img, label
batch_size  = 128
train_dataset = FlowerDataSet(train_file_labels)
train_loader = DataLoader(dataset=train_dataset,  #打包
                          batch_size=batch_size,
                          shuffle=True)
test_dataset = FlowerDataSet(test_file_labels)
test_loader = DataLoader(dataset=test_dataset,  #打包
                          batch_size=batch_size,
                          shuffle=True)
# ============= 以下构建模型 =============================
model = EfficientNet.from_pretrained('efficientnet-b7').to(device)
......  # 【此处为本例核心代码，已省略，完整代码见教材P143；建议读者手工输入核心代码并进行调试，这样方能领会其含义】



class EfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b3_ns', pretrained=True):
        super().__init__()
        self.model = model   #构建efficientnet模型，使用预训练模型
        self.fc1 = fc1
        self.fc2 = fc2
    def forward(self, x):
        o = x
        o = self.model(o)

        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)

        o = self.fc1(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)

        o = self.fc2(o)

        return o

efficient_model = EfficientNet().to(device)
optimizer = optim.Adam(efficient_model.parameters())

# ============= 以下开始训练和测试 ================
#给定数据集，测试在其上的准确率：
def getAccOnadataset(data_loader):
    efficient_model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pre_y = efficient_model(x)
            pre_y = torch.argmax(pre_y, dim=1)
            t = (pre_y == y).long().sum()
            correct += t
        correct = 1. * correct / len(data_loader.dataset)
    efficient_model.train()
    return correct.item()

start=time.time() 		    #开始计时
efficient_model.train()
for epoch in range(30): 	#执行30代
    ep_loss=0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = efficient_model(x)
        loss = nn.CrossEntropyLoss()(pre_y, y)
        print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#训练结束
end=time.time()
print('运行时间：',(end-start)/60.,'分钟')
torch.save(efficient_model,'efficient_model')
acc_test = getAccOnadataset(test_loader)
print('在测试集上的准确率：',acc_test)





'''
19 0.15419840812683105
19 0.8684183955192566
运行时间： 11.100697712103527 分钟
在测试集上的准确率： 0.856494128704071


'''





