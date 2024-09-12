import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os #cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判断是否有GPU
# -----------------------------------------------
batch_size = 512

def tfm(x):
    x = tfs.ToTensor()(x)
    return  (x - 0.5) / 0.5
train_set = MNIST(root='./data/mnist2',train=True,download=True,transform=tfm)
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self,noise_len, h,w): #h,w分别为图像的高和宽，单通道图像
        super().__init__()
        self.h = h
        self.w = w
        self.fc = nn.Sequential(
            nn.Linear(noise_len, 2048),
            nn.ReLU(True),
            ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P173；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

                nn.Tanh(),
            )
    def forward(self, x):
        o = x
        o = self.fc(o) #torch.Size([512, 784])
        o = o.reshape(x.shape[0],1,self.h,self.w) #torch.Size([512, 1, 28, 28])
        return o

#定义辨识器类
class Discriminator(nn.Module):
    def __init__(self,h,w): #h,w分别为图像的高和宽，单通道图像
        super().__init__()
        self.fc = nn.Sequential(
            ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P174；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

                nn.Linear(256, 1),
        )
    def forward(self, x): #torch.Size([512, 1, 28, 28])
        o = x
        o = o.reshape(x.shape[0],-1)
        o = self.fc(o)
        return o


noise_len = 96  #噪声的维度
discriminator = Discriminator(28,28).to(device)
generator = Generator(noise_len,28,28).to(device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=5e-4, betas=(0.5, 0.999))

for ep in range(100):
    for k, (img, _) in enumerate(train_data):  # x==torch.Size([128, 1, 28, 28])
        real_img = img.to(device)
        size = real_img.shape[0]
        one_labels = torch.ones(size, 1).float().to(device)
        zero_labels = torch.zeros(size, 1).float().to(device)
        # 训练辨识网络
        score_real = discriminator(real_img) #torch.Size([512, 1])
        noise_data = torch.rand(size, noise_len).to(device)   # 生成噪声数据,torch.Size([512, 96])
        noise_data = (noise_data-0.5)/0.5
        fake_img = generator(noise_data)  ##torch.Size([512, 1, 28, 28])
        score_fake = discriminator(fake_img)
        #希望识别器能够识别真图和假图
        d_loss = nn.BCEWithLogitsLoss()(score_real, one_labels) \
                 + nn.BCEWithLogitsLoss()(score_fake, zero_labels)
        #希望真图像的辨别分值（概率值）接近于1
        #希望假图像的辨别分值（概率值）接近于0
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()  # 优化判别网络
        #-------------------------------------
        # 训练生成网络
        noise_data = torch.rand(size, noise_len).to(device)  # 生成噪声数据,torch.Size([512, 96])
        noise_data = (noise_data - 0.5) / 0.5
        fake_img = generator(noise_data)
        score_fake = discriminator(fake_img)
    
        g_loss = nn.BCEWithLogitsLoss()(score_fake,one_labels) #希望假的图像尽可能逼真
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if k % 20 == 0:
            print(ep, d_loss.item(), g_loss.item())

#-----训练结束-----
torch.save(generator,'generator')

size = 16

generator = torch.load('generator').to(device)
noise_data = torch.rand(size, noise_len).to(device)  # 生成噪声数据,torch.Size([512, 96])
noise_data = (noise_data-0.5)/0.5
fake_img = generator(noise_data)
imgs = np.array(fake_img.data.cpu())

_, axu = plt.subplots(4, 4, figsize=(4, 4))
#plt.ion()
row_img_n = 4 #一行上显示的图像数
for i in range(row_img_n*row_img_n):
    axu[i // row_img_n][i % row_img_n].imshow(np.reshape(imgs[i], (28, 28)), cmap='gray')
    axu[i // row_img_n][i % row_img_n].set_xticks(())
    axu[i // row_img_n][i % row_img_n].set_yticks(())

plt.suptitle('The 16 handwritten digits')
plt.show() #pause(1)
exit(0)

