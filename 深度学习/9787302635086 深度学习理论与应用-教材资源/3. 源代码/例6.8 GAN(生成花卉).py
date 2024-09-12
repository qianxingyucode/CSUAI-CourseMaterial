import numpy as np
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = datasets.ImageFolder(
    root="./data/flower_dataset",  # 这种方式读取文件时，root指定的目录下必须有且仅有一级子目录，每个子目录就是一个类别
    transform=transforms.Compose([
        transforms.Resize((64 ,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
)
data_loader = DataLoader(dataset=dataset ,batch_size=128 ,shuffle=True)
'''
#使用高斯分布初始化参数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
'''
# 定义生成器类
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cnn = nn.Sequential(  #利用反卷积来生成图像
            nn.ConvTranspose2d(100, 512, kernel_size=(4, 4)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            #上面语句等价于：nn.ConvTranspose2d(512, 1024, 4, 2, 1)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P176；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

            nn.ConvTranspose2d(512, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )
    def forward(self, x):
        o = self.cnn(x)
        return o
# 定义判别器类
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
                   nn.Conv2d(3, 64, 4, 2, 1),
                   nn.LeakyReLU(0.2),
                   nn. Conv2d(64, 128, 4, 2, 1),
                   nn.BatchNorm2d(128),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(128, 256, 4, 2, 1),
                   nn.BatchNorm2d(256),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(256, 128, 4, 2, 1),
                   ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P177；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        )
    def forward(self, x):
        o = self.cnn(x)
        o = o.squeeze()
        return o

generator = Generator().to(device)
discriminator = Discriminator().to(device)

#generator.apply(weights_init_normal)  #
#discriminator.apply(weights_init_normal) #
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))



EPOCH = 500
for epoch in range(EPOCH):
    for i, (imgs, _) in enumerate(data_loader):
        imgs = imgs.to(device)
        # valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        one_labels = torch.ones(imgs.size(0)).to(device)
        zero_labels = torch.zeros(imgs.size(0)).to(device)

        # 训练生成器
        g_optimizer.zero_grad()
        z = torch.randn(imgs.size()[0], 100, 1, 1).to(device) #生成噪声
        fake_imgs = generator(z) #生成假图
        fake_scores = discriminator(fake_imgs)
        g_loss = torch.nn.BCELoss()(fake_scores, one_labels )# 计算生成器的损失函数值
        g_loss.backward()
        g_optimizer.step()  #更新生成器的参数

        #训练判别器
        d_optimizer.zero_grad()
        real_imgs = imgs
        real_scores = discriminator(real_imgs)
        real_loss = torch.nn.BCELoss()(real_scores , one_labels) #使得辨识器能够分辨真图
        fake_scores = discriminator(fake_imgs.detach())
        fake_loss = torch.nn.BCELoss()(fake_scores, zero_labels) #使得辨识器能够分辨假图
        d_loss = real_loss + fake_loss  #使得辨识器能够同时分辨真图和假图
        d_loss.backward()
        d_optimizer.step()  #更新辨识器的参数

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCH, i, len(data_loader), d_loss.item(), g_loss.item())
        )
'''
'''

torch.save(generator ,'flower_generator')
exit(0)
generator = torch.load('flower_generator').to(device)
generator.eval()
z = torch.randn(128, 100, 1, 1).to(device)  # torch.Size([128, 100, 1, 1])
gen_imgs = generator(z)
save_image(gen_imgs.data[:16], "sample.png", nrow=4, normalize=True)


'''
[Epoch 499/500] [Batch 4/11] [D loss: 0.040216] [G loss: 4.615952]
[Epoch 499/500] [Batch 5/11] [D loss: 0.037123] [G loss: 4.543477]
[Epoch 499/500] [Batch 6/11] [D loss: 0.046811] [G loss: 4.313377]
[Epoch 499/500] [Batch 7/11] [D loss: 0.047385] [G loss: 4.604750]
[Epoch 499/500] [Batch 8/11] [D loss: 0.040389] [G loss: 4.330332]
[Epoch 499/500] [Batch 9/11] [D loss: 0.035349] [G loss: 4.738313]
[Epoch 499/500] [Batch 10/11] [D loss: 0.041240] [G loss: 4.758043]
-------


[Epoch 499/500] [Batch 6/11] [D loss: 0.110900] [G loss: 4.374277]
[Epoch 499/500] [Batch 7/11] [D loss: 0.198603] [G loss: 2.674938]
[Epoch 499/500] [Batch 8/11] [D loss: 0.212808] [G loss: 2.560666]
[Epoch 499/500] [Batch 9/11] [D loss: 0.203449] [G loss: 3.876668]
[Epoch 499/500] [Batch 10/11] [D loss: 0.214477] [G loss: 2.787088]


'''