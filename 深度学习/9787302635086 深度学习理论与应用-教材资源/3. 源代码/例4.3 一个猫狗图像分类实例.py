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
    #transforms.RandomCrop(224,224),#再随机载剪至
    #transforms.RandomHorizontalFlip(), #随机水平翻转图像
    transforms.ToTensor(),  #转化张量
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #归一化
])

class cat_dog_dataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = os.listdir(dir)
    def __len__(self): #调用DataLoader的实例的DataLoader属性时会用到该方法
        t = len(self.files)  #文件个数，样本条数
        return t
    def __getitem__(self, idx):
        file = self.files[idx]
        fn = os.path.join(self.dir, file)
        img = Image.open(fn).convert('RGB')
        img = transform(img)  # 调整大小 转为张量
        img = img.reshape(224,224,-1)
        y = 0 if 'cat' in file else 1
        return img,y
#=============================================
batch_size  = 20
train_dir = './data/catdog/training_set'  #训练集所在的目录，猫和狗的图像文件混在一起
test_dir ='./data/catdog/test_set'        #测试集所在的目录，猫和狗的图像文件混在一起

train_dataset = cat_dog_dataset(train_dir)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = cat_dog_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True)

print('训练集大小：',len(train_loader.dataset))
#================================
#定义卷积神经网络
class Model_CatDog(nn.Module):
    def __init__(self):
        super().__init__()

        #以下定义4个卷积层：
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2) #5*5卷积核，默认步长1，填充2
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P113；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        # 以下定义3个全连接层：
        self.fc1 = nn.Linear(256*12*12, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self,x): # torch.Size([20, 224, 224, 3])
        x = x.permute([0,3,1,2])  #形状改变：(batch, 224, 224, 3)--->(batch, 3, 224, 224)

        ......  # 【此处为本例核心代码，已省略，完整代码见教材P113；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        out = nn.Dropout(0.5)(out)
        out = self.fc2(out)               #(batch, 2048)--->(batch, 512)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc3(out)               #(batch, 512)--->(batch, 2)

        return out #(batch, 2)
#---------------------------------------------
model_CatDog = Model_CatDog().to(device)
optimizer = torch.optim.SGD(model_CatDog.parameters(), lr=0.01, momentum=0.9)
start=time.time() #开始计时
model_CatDog.train()
for epoch in range(30): #执行30代
    ep_loss=0
    for i,(x,y) in enumerate(train_loader):
        x, y = x.to(device),y.to(device)
        pre_y = model_CatDog(x)
        loss = nn.CrossEntropyLoss()(pre_y, y.long())  # 使用交叉熵损失函数
        ep_loss += loss*x.size(0)  #loss是损失函数的平均值，故要乘以样本数量
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('第 %d 轮循环中，损失函数的平均值为: %.4f'\
          %(epoch+1,(ep_loss/len(train_loader.dataset))))


end = time.time() #计时结束
print('训练时间为:  %.1f 秒 '%(end-start))

#=============================================
torch.save(model_CatDog,'model_CatDog')
model_CatDog = torch.load('model_CatDog')
correct = 0
model_CatDog.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = model_CatDog(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t

t = 1.*correct/len(train_loader.dataset)
print('1、网络模型在训练集上的准确率：{:.2f}%'\
      .format(100*t.item()))

correct = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        pre_y = model_CatDog(x)

        pre_y = torch.argmax(pre_y, dim=1)

        t = (pre_y == y).long().sum()

        correct += t
t = 1.*correct/len(test_loader.dataset)
print('2、网络模型在测试集上的准确率：{:.2f}%'\
      .format(100*t.item()))

'''
29 0.0008197164279408753
29 0.018392646685242653
训练集上的准确率： 0.9902560710906982
测试集上的准确率： 0.8344042897224426
训练时间为:  1381.5 秒 
训练时间为:  1371.9 秒 (用了序列结构)

'''

exit(0)





exit(0)



