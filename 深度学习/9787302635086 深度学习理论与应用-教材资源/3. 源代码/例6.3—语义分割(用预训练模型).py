import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有GPU
# ---------------------------------------------------------------------

torch.manual_seed(123)
random.seed(123)
np.random.seed(10)


class OneModule(nn.Module):
    def __init__(self, n1, n2): # n1, n2分别为输入和输出特征图的通道数
        super(OneModule, self).__init__()
        self.cnn = nn.Sequential(  # 卷积模块类
            nn.Conv2d(n1, n2, 3, padding=1, bias=False),
            nn.BatchNorm2d(n2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n2, n2, 3, padding=1, bias=False),
            nn.BatchNorm2d(n2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        o = self.cnn(x)
        return o


class UNet(nn.Module):
    def __init__(self, n1, n2):
        super(UNet, self).__init__()
        self.cnn1 = OneModule(n1, 64)
        self.cnn2 = OneModule(64, 128)
        self.cnn3 = OneModule(128, 256)
        self.cnn4 = OneModule(256, 512)
        self.bottleneck = OneModule(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ucnn4 = OneModule(1024, 512)
        self.ucnn3 = OneModule(512, 256)
        self.ucnn2 = OneModule(256, 128)
        self.ucnn1 = OneModule(128, 64)
        self.contr4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.contr3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.contr2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.contr1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1_1 = nn.Conv2d(64, 1, kernel_size=1)  # 1*1卷积，用于降维
    def forward(self, x): #torch.Size([16, 3, 160, 240])
        skip_cons = []
        d1 = x
        d1 = self.cnn1(d1)  # (3, 160, 240)-->(64, 160, 240),特征图的通道数改变（n1--->n2），长和宽不变
        skip_cons.append(d1)  # 保存本次卷积结果，位于skip_cons[0]
        d1 = self.pool(d1)  # (64, 160, 240)-->(64, 80, 120)   通道数不变，长和宽减半
        d2 = d1
        d2 = self.cnn2(d2)  # (64, 80, 120) -->(128, 80, 120)
        skip_cons.append(d2)  # 位于skip_cons[1]
        d2 = self.pool(d2)  # (128, 80, 120) -->(128, 40, 60)
        d3 = d2
        d3 = self.cnn3(d3)  # (128, 40, 60) -->(256, 40, 60)
        skip_cons.append(d3)  # 位于skip_cons[2]
        d3 = self.pool(d3)  # (256, 40, 60) -->(256, 20, 30)
        d4 = d3
        d4 = self.cnn4(d4)  # (256, 20, 30) -->(512, 20, 30)
        skip_cons.append(d4)  # 位于skip_cons[3]
        d4 = self.pool(d4)  # (512, 20, 30) -->(512, 10, 15)
        # =======================
        bo = d4
        bo = self.bottleneck(bo)  # (512, 10, 15) -->(1024, 10, 15)
        # =======================
        u4 = bo
        u4 = self.contr4(u4)  # (1024, 10, 15)-->(512, 20, 30),通道数减半，特征图尺寸增加一倍
        # (512, 20, 30)+(512, 20, 30)-->(1024, 20, 30)
        u4 = torch.cat((skip_cons[3], u4), dim=1)
        u4 = self.ucnn4(u4)  # (1024, 20, 30)-->(512, 20, 30)
        u3 = u4
        u3 = self.contr3(u3)  # (512, 20, 30)-->(256, 40, 60)(省了参数名，下同)
        # (256, 40, 60)+(256, 40, 60)-->(512, 40, 60)
        u3 = torch.cat((skip_cons[2], u3), dim=1)
        u3 = self.ucnn3(u3)  # (512, 40, 60)-->(256, 40, 60)
        u2 = u3
        u2 = self.contr2(u2)  # (256, 40, 60)-->(128, 80, 120)(省了参数名，下同)
        # (128, 80, 120)+(128, 80, 120)-->(256, 80, 120)
        u2 = torch.cat((skip_cons[1], u2), dim=1)
        u2 = self.ucnn2(u2)  # (256, 80, 120)-->(128, 80, 120)
        u1 = u2
        u1 = self.contr1(u1)  # (128, 80, 120)-->(64, 160, 240)
        # (64, 160, 240)+(64, 160, 240))-->(128, 160, 240)
        u1 = torch.cat((skip_cons[0], u1), dim=1)
        u1 = self.ucnn1(u1)  # (128, 160, 240)-->(64, 160, 240)
        o = self.conv1_1(u1)  # (64, 160, 240)-->(1, 160, 240)
        return o


class GetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, flag):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = os.listdir(img_dir)
        self.flag = flag
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        mask_path = os.path.join(self.mask_dir, self.imgs[index].replace(".jpg", "_mask.gif"))
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img = np.array(img)
        mask = np.array(mask)
        if self.flag == 'train':
            img = train_transform(img)
        elif self.flag == 'test':
            img = test_transform(img)
        else:
            print('Error！')
            exit(0)
        mask = test_transform(mask)
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        return img, mask
# =================================================================
train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((160, 240)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
test_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((160, 240)),
     transforms.ToTensor()]
)
# --------------------------------------------------------
unet_model = UNet(n1=3, n2=1).to(device)
model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

......  # 【此处为本例部分核心代码，已省略，完整代码见教材P161；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

model = model.to(device)
for param in model.parameters():
    param.requires_grad = False   #先冻结所有参数
for param in model.classifier.parameters():
    param.requires_grad = True    #放开分类器的参数，使之可训练
unet_model = model


optimizer = optim.Adam(unet_model.parameters(), lr=1e-4)
train_dataset = GetDataset(img_dir='./data/semantic-seg/train_imgs/',
                           mask_dir='./data/semantic-seg/train_masks/', flag='train')
train_loader = DataLoader(train_dataset,batch_size=16,num_workers=0,pin_memory=True,shuffle=True)

for ep in range(10):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.float().to(device)
        pre_y = unet_model(x)['out']
        loss = nn.BCEWithLogitsLoss()(pre_y, y)
        if batch_idx%10 == 0:
            print(ep, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(unet_model,'unet_model_pre')

#------------------------------------------------------
def showTwoimgs(imgs, stitle='', rows=1, cols=2):
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for idx, title in enumerate(imgs):
        ax.ravel()[idx].imshow(imgs[title])
        ax.ravel()[idx].set_title(title)
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    plt.suptitle(stitle, fontsize=18, color='red')
    plt.show()
unet_model = torch.load('unet_model_pre')  # unet_model10 是没有被覆盖的
unet_model.eval()
val_dataset = GetDataset(img_dir='./data/semantic-seg/val_imgs/',
                         mask_dir='./data/semantic-seg/val_masks/', flag='test')
val_loader = DataLoader(val_dataset,batch_size=16,num_workers=0,pin_memory=True,shuffle=True)
num_correct = 0
num_pixels = 0
dice_score = 0
with torch.no_grad():
    for i,(x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        pre_y = unet_model(x)['out']
        pre_y = torch.sigmoid(pre_y)  # 映射到(0, 1)之间
        #计算指标Dice score
        mask = y
        del y
        pre_mask = (pre_y >= 0.1).float()  #对预测输出的掩码图像进行二值化
        num_correct += (pre_mask == mask).sum()
        num_pixels += torch.numel(pre_mask)
        tmp = (2 * (pre_mask * mask).sum()) / ((pre_mask + mask).sum() + 1e-8)  #计算批内的Dice score平均值
        dice_score += tmp
        #print(i, tmp.item())
        pre_mask = pre_mask[0, 0].cpu() #选择一副掩码图来对比
        mask = mask[0, 0].cpu()
        imgs = dict()
        imgs['Original mask'] = np.array(mask) #事先标注的语义掩码图像
        imgs['Predictive mask'] = np.array(pre_mask) #模型预测时输出的语义掩码图像
        showTwoimgs(imgs, '', 1, 2) #该函数代码见例??

print(f"准确率为： {1.*num_correct}/{num_pixels} = {1.*num_correct / num_pixels * 100:.2f}%")
print(f"指标Dice score的值为: {1.*dice_score / len(val_loader):.2f}")


