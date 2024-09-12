import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import random
import time
from vit_model import vit_base_patch16_224_in21k as create_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_model().to(device)
......  # 【此处为本例部分核心代码，已省略，完整代码见教材P262；建议读者手工输入核心代码并进行调试，这样方能领会其含义】



class ViTforCifar100(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(21843, 1024) 	#第1个全连接层
        self.fc2 = nn.Linear(1024, 100)		#第2个全连接层
    def forward(self,x): 		#torch.Size([32, 3, 224, 224])
        out = self.model(x) 	#torch.Size([32, 21843])
        out = torch.relu(out)
        #以下对model的输出做微调
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)		#输出张量的形状为#torch.Size([32, 100])
        return out
dataset = datasets.ImageFolder(
    root="../data/cifar100",        		#利用指定目录读取数据
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),   	#调整为224*224*3尺寸的图像
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
)
#以下代码将数据集对象划分为训练集和测试集，最后分别打包
sample_indices = list(range(len(dataset)))
random.shuffle(sample_indices) 			#打乱样本的索引顺序
train_len = int(0.7*len(sample_indices))	#确定训练集的长度
train_indices = sample_indices[:train_len]	#训练集样本的索引
test_indices = sample_indices[train_len:]	#测试集样本的索引
#利用索引来构建数据集对象
train_set = torch.utils.data.Subset(dataset, train_indices)	#Subset类型
test_set = torch.utils.data.Subset(dataset, test_indices)
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)  #训练集
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)    #测试集

vit_cifar100_model = ViTforCifar100().to(device)
optimizer = optim.Adam(vit_cifar100_model.parameters())
start=time.time()
for epoch in range(20):      #迭代20代
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        pre_imgs = vit_cifar100_model(imgs) 	#torch.Size([32, 100])
        loss = nn.CrossEntropyLoss()(pre_imgs, labels)  # 使用交叉熵损失函数
        print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

end = time.time()
torch.save(vit_cifar100_model,'vit_cifar100_model')
print('time cost(耗时)：%0.2f 分钟'% ((end - start)/60.0))






