import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
from torchvision import datasets, transforms, models
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============= 以下开始读数据并打包 =====================
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
file_labels = getFileLabel(path)  #获取(文件名,类别编号)格式组成的list，总数：3670
random.shuffle(file_labels)  #打乱顺序
random.shuffle(file_labels)
#按7:3划分训练集和测试集：
rate = 0.7
train_length = int(rate*len(file_labels))
train_file_labels = file_labels[:train_length]
test_file_labels = file_labels[train_length:]
#train_file_labels,test_file_labels  = train_file_labels[0:150],test_file_labels[0:150]  #暂时缩短
transform = transforms.Compose([
    transforms.Resize((224,224)),  #调整图像大小为(224,224)
    transforms.ToTensor(),  #转化张量
])
class FlowerDataSet(Dataset):               #定义数据集类
    def __init__(self, data_file_label):
        self.data_file_label = data_file_label
    def __len__(self):     #需要重写该方法，返回数据集大小
        t = len(self.data_file_label)
        return t
    def __getitem__(self, idx):
        fn, label = self.data_file_label[idx][0], self.data_file_label[idx][1]
        img = Image.open(fn).convert('RGB')
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
'''
googlenet_base = models.googlenet(num_classes=5,init_weights=True) #下载模型需要时间，笔者已经将模型下载并放在指定的目录了
torch.save(googlenet_base,'./pre_models/googlenet_base5')
'''
googlenet_base = torch.load('./pre_models/googlenet_base5') #加载已下载的模型，类别个数为5


model_dict = googlenet_base.state_dict()
pretrain_model = torch.load(f"./pre_models/googlenet-1378be20.pth")  #加载最新的模型参数

......  # 【此处为本例核心代码，已省略，完整代码见教材P139；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

#----------------
for param in googlenet_base.parameters(): #先冻结所有的参数
    param.requires_grad = False
#解冻五个全连接层：
layers = [googlenet_base.aux1.fc1,googlenet_base.aux1.fc2,googlenet_base.aux2.fc1,googlenet_base.aux2.fc2,googlenet_base.fc]
for layer in layers:
    for param in layer.parameters():
        param.requires_grad = True

'''
for i,param in enumerate(googlenet_base.parameters()): #查看冻结情况
    print(i + 1, param.requires_grad)
exit(0)
'''
googlenet_model = googlenet_base.to(device)
optimizer = optim.Adam(googlenet_model.parameters())
# ============= 以下开始训练和测试 ================
#给定数据集，测试在其上的准确率：
def getAccOnadataset(data_loader):
    googlenet_model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pre_y = googlenet_model(x) #在训练模式下，pre_y的类型为torch.Tensor
            #pre_y = pre_y.logits #在训练模式下，不需要
            pre_y = torch.argmax(pre_y, dim=1)
            t = (pre_y == y).long().sum()
            correct += t
        correct = 1. * correct / len(data_loader.dataset)
    googlenet_model.train()
    return correct.item()
start=time.time() 		#开始计时
googlenet_model.train()
for epoch in range(10): 	#执行10代
    ep_loss=0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = googlenet_model(x)
        pre_y = pre_y.logits
        loss = nn.CrossEntropyLoss()(pre_y, y)
        print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#训练结束
end=time.time() #计时结束
print('运行时间：',(end-start)/60.,'分钟')
torch.save(googlenet_model,'googlenet_model')

acc_test = getAccOnadataset(test_loader)
print('在测试集上的准确率：',acc_test)


'''
#训练所有参数
9 0.08904056996107101
9 0.3456568419933319
运行时间： 2.056525417168935 分钟
在测试集上的准确率： 0.8964577913284302

#训练部分参数
9 0.4279835522174835
9 0.22218433022499084
运行时间： 1.9098375598589579 分钟
在测试集上的准确率： 0.8746594190597534
'''
