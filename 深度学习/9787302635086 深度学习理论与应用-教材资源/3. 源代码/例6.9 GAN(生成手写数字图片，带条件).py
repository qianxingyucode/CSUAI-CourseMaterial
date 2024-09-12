import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100 #噪声向量的长度
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 32)
        self.fc = nn.Sequential(
            nn.Linear(132, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P180；建议读者手工输入核心代码并进行调试，这样方能领会其含义】


        )
    def forward(self, noise, labels): #torch.Size([128, 100]) torch.Size([128])
        label_emb = self.label_emb(labels) #torch.Size([128, 10])
        gen_input = torch.cat((label_emb, noise), -1) #torch.Size([128, 10]) torch.Size([128, 100])-->torch.Size([128, 110])
        img = self.fc(gen_input) #torch.Size([128, 784])
        img = img.reshape(img.size(0),-1,28,28)
        return img
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 32)
        self.model= nn.Sequential(
    nn.Linear(816, 512),
    nn.LeakyReLU(0.2, True),
    nn.Linear(512, 512),
    nn.Dropout(0.4, False),
    ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P180；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    nn.LeakyReLU(0.2, True),
    nn. Linear(512, 1),
  )
    def forward(self, img, labels):
        img = img.reshape(img.size(0), -1)
        label_emb = self.label_embedding(labels)
        # torch.Size([128, 784])+torch.Size([128, 10])---> torch.Size([128, 794])  816
        img_label = torch.cat((img, label_emb), -1)
        out = self.model(img_label)  #torch.Size([128, 1])
        return out
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#-------------------------------------------------
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist",train=True,download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=128,
    shuffle=True,
)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

EPOCH = 500
for epoch in range(EPOCH):
    for i, (imgs, labels) in enumerate(dataloader):
        one_labels = torch.ones(imgs.size(0)).to(device)
        zero_labels = torch.zeros(imgs.size(0)).to(device)
        real_imgs = torch.FloatTensor(imgs).to(device)
        labels = torch.LongTensor(labels).to(device)
        #训练生成器
        g_optimizer.zero_grad()
        z = torch.randn(imgs.size(0),  z_dim).to(device)  # 生成噪声
        gen_labels = torch.randint(0, 10, [imgs.size(0)]).to(device)  #随机生成类编号,一共有10个类
        gen_imgs = generator(z, gen_labels)  #利用噪声和类编号生成假图
        gen_scores = discriminator(gen_imgs, gen_labels).squeeze()
        g_loss = nn.MSELoss()(gen_scores, one_labels)  #希望生成器生成尽可能逼真的图（假图）
        g_loss.backward()
        g_optimizer.step()  #更新生成器参数
        # 训练辨识器
        d_optimizer.zero_grad()
        real_scores = discriminator(real_imgs, labels).squeeze() #对真实图像打分
        d_real_loss = nn.MSELoss()(real_scores, one_labels) #希望辨识器对真实图的打分是正确的
        fake_scores = discriminator(gen_imgs.detach(), gen_labels).squeeze() #对假图打分
        d_fake_loss = nn.MSELoss()(fake_scores, zero_labels)   #希望辨识器对假图的打分是正确的
        d_loss = (d_real_loss + d_fake_loss) / 2 #希望辨识器对真土和假图的判断尽可能正确
        d_loss.backward()
        d_optimizer.step()
        if i%100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCH, i, len(dataloader), d_loss.item(), g_loss.item())
            )
torch.save(generator,'cond_generator2')
'''
'''
generator = torch.load('cond_generator2').to(device)
generator.eval()
digit = 8 # 要生成的数字
z = torch.randn(16,  z_dim).to(device)  # 生成噪声
gen_labels = torch.LongTensor(np.array([digit for _ in range(16)])).to(device)
#gen_labels = torch.randint(0,10,[16]).to(device)
gen_imgs = generator(z, gen_labels)
save_image(gen_imgs.data, "sample.png", nrow=4, normalize=True)
