import random
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义数据集类
class RoadDataset(Dataset):
    def __init__(self, paths, bb, y):
        self.paths = paths.values   	#图像的路径
        self.bb = bb.values        	#各图像标注框box的坐标，与图像一一对应
        self.y = y.values          	#图像的类别编号
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        img = cv2.imread(str(path)).astype(np.float32)  #读取指定的图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255  #归一化
        y_bb = self.bb[idx]  #读取box的坐标，包含四个数字，
        #前面两个为box的左上角坐标，后两个为右下角坐标
        tmplist=y_bb.split(' ')
        y_bb = [int(e) for e in tmplist]
        y_bb = torch.tensor(y_bb)
        img = torch.Tensor(img).permute([2,0,1])
        return img, y_class, y_bb

df_train=pd.read_csv('./data/object_detection/dataset.csv')
X = df_train[['new_path', 'new_bb']]  #包含图片的路径以及box的坐标
Y = df_train['class']               # 类别标签
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
random_state=42, shuffle=False)  # 划分为训练集和测试集，大小分别为701和176
train_ds = RoadDataset(X_train['new_path'], X_train['new_bb'], y_train)
test_ds = RoadDataset(X_test['new_path'], X_test['new_bb'], y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)


#定义目标检测网络
class Detect_model(nn.Module):
    def __init__(self):
        super(Detect_model, self).__init__()
        resnet = models.resnet34(pretrained=True) #导入resnet34
        layers = list(resnet.children())[:8] 		#取resnet的前八层
        self.features = nn.Sequential(*layers)	#用于图像的特征提取
#图像分类网络，有4个类别
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
#box坐标的预测网络，有4个参数需要预测
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
    def forward(self, x):
        o = x
        o = self.features(o)
        o = torch.relu(o)
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P163；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        return self.classifier(o), self.bb(o)
detect_model = Detect_model().to(device)
parameters = filter(lambda p: p.requires_grad, detect_model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.001)
#-------------- 以下开始训练 ----------------
detect_model.train()
for ep in range(200):
    for k,(x, y_class, y_bb) in enumerate(train_loader):
        x, y_class, y_bb = x.to(device), y_class.to(device).long(), y_bb.to(device).float()
        pre_y, pre_bb = detect_model(x)
        loss_class = F.cross_entropy(pre_y, y_class, reduction="sum")
        loss_bb = F.l1_loss(pre_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        #loss_bb是的1000倍左右，为了平衡，故乘以0.001
        loss = loss_class + 0.001*loss_bb
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(detect_model, 'detect_model')  #保存模型
#-------------- 以下开始测试 ----------------
class_names = {0:'speedlimit', 1:'stop', 2: 'crosswalk', 3: 'trafficlight'}  # 四个类别
detect_model = torch.load('detect_model')
detect_model.eval()
correct = 0
for k,(x, y_class, y_bb) in enumerate(test_loader):
    x, y_class, y_bb = x.to(device), y_class.to(device).long(), y_bb.to(device).float()
    pre_y, pre_bb = detect_model(x) #torch.Size([16, 4]) torch.Size([16, 4])
    _, pre_index = torch.max(pre_y, 1)
    t = (pre_index == y_class).int().sum()  #计算准确率
    correct += t
    '''
    #以下显示目标检测效果及分类效果
    img = x[0].permute([1,2,0]).cpu()
    img = np.array(img)
    img = img.copy()  # 原因不明
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    name_index = pre_index[0].item() #获取类名的编号
    label = pre_bb[0].long()
    cv2.rectangle(img, (label[1], label[0]), (label[3], label[2]), color=(255, 0, 0), thickness=2)
    cv2.putText(img, text=str(class_names[name_index]), org=(label[1], label[0] - 5),
 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1,
                lineType=cv2.LINE_AA, color=(0, 0, 255))
    cv2.imshow('11', img)
    cv2.waitKey(0)    
    '''
correct = 1.*correct/len(test_loader.dataset)
print('在测试集上的分类准确率为：{:.2f}%'.format(100*correct.item()))
