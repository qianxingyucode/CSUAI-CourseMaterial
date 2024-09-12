from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-------------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),  #调整图像大小为(224,224)
    transforms.ToTensor(),  #转化张量
])
class cat_dog_dataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = os.listdir(dir)
    def __len__(self):     #需要重写该方法，返回数据集大小
        t = len(self.files)
        return t
    def __getitem__(self, idx):
        file = self.files[idx]
        fn = os.path.join(self.dir, file)
        img = Image.open(fn).convert('RGB')
        img = transform(img)       #调整图像形状为(3,224,224), 并转为张量
        img = img.reshape(-1,224,224)
        y = 0 if 'cat' in file else 1     #构造图像的类别
        return img,y
#=============================================
batch_size  = 20
train_dir = './data/catdog/training_set2'   #训练集所在的目录
test_dir ='./data/catdog/test_set'        #测试集所在的目录
train_dataset = cat_dog_dataset(train_dir)        #创建数据集
train_loader = DataLoader(dataset=train_dataset,  #打包
                          batch_size=batch_size,
                          shuffle=True)
test_dataset = cat_dog_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True)
print('训练集大小：',len(train_loader.dataset))
print('测试集大小：',len(test_loader.dataset))
#================================
cat_dog_vgg16 = models.vgg16(pretrained=True).to(device)

......  # 【此处为本例核心代码，已省略，完整代码见教材P120；建议读者手工输入核心代码并进行调试，这样方能领会其含义】


cat_dog_vgg16.train()



cat_dog_vgg16 = cat_dog_vgg16.to(device)
optimizer = torch.optim.SGD(cat_dog_vgg16.parameters(), lr=0.01, momentum=0.9)

start=time.time() #开始计时
cat_dog_vgg16.train()
for epoch in range(10): #执行10代
    ep_loss=0
    for i,(x,y) in enumerate(train_loader):
        x, y = x.to(device),y.to(device)
        pre_y = cat_dog_vgg16(x)
        loss = nn.CrossEntropyLoss()(pre_y, y.long())  # 使用交叉熵损失函数
        ep_loss += loss*x.size(0) #loss是损失函数的平均值,故要乘以样本数量
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    print('第 %d 轮循环中，损失函数的平均值为: %.4f'\
          %(epoch+1,(ep_loss/len(train_loader.dataset))))
end = time.time() #计时结束
print('训练时间为:  %.1f 秒 '%(end-start))
#=============================================
correct = 0
cat_dog_vgg16.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(train_loader):#计算在训练集上的准确率
        x, y = x.to(device), y.to(device)
        pre_y = cat_dog_vgg16(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t
t = 1.*correct/len(train_loader.dataset)
print('1、网络模型在训练集上的准确率：{:.2f}%'\
      .format(100*t.item()))

correct = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):#计算在测试集上的准确率
        x, y = x.to(device), y.to(device)
        pre_y = cat_dog_vgg16(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t
t = 1.*correct/len(test_loader.dataset)
print('2、网络模型在测试集上的准确率：{:.2f}%'\
      .format(100*t.item()))
