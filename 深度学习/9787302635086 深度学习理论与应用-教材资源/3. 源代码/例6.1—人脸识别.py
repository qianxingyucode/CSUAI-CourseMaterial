import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision
import random
import time
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------
torch.manual_seed(123)
random.seed(123)
np.random.seed(10)

# ===============================================================
dir_path = './data/faces/training'


# 训练数据集获取
# 获取path2目录下所有的文件名（含路径）和类别目录编号，然后返回以二元组(文件名,类别目录编号)为元素的list
def getFn_Dir(tpath):
    dirs = os.listdir(tpath)  # 获得所有类别目录名
    file_labels = []  # 用于保存(文件名,类别目录编号)的list
    for i, dir in enumerate(dirs):
        label = i  # 按目录对类别编号
        path2 = os.path.join(tpath, dir)
        files = os.listdir(path2)  # 获取当前类别目录下的所有文件名
        for file in files:
            fn = os.path.join(path2, file)  # 具体的文件名（含路劲）
            t = (fn, label)
            file_labels.append(t)
    random.shuffle(file_labels)
    random.shuffle(file_labels)
    random.shuffle(file_labels)
    return file_labels


transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((100, 100)), transforms.RandomHorizontalFlip(p=0.5),
     transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# 生成人脸数据集类
class FaceDataset(Dataset):
    def __init__(self, fn_labels2):
        self.fn_labels = fn_labels2

    def __getitem__(self, idx):
        img1, label1 = self.fn_labels[idx]
        fg = random.randint(0, 1)  # 随机生成0或1
        if fg == 1:  # 生成同类的三元组
            k = idx + 1
            while True:
                if k >= len(self.fn_labels):
                    k = 0
                img2, label2 = self.fn_labels[k]
                k += 1
                if int(label1) == int(label2):
                    break
        else:  # 生成不同类的三元组
            k = idx + 1
            while True:
                if k >= len(self.fn_labels):
                    k = 0
                img2, label2 = self.fn_labels[k]
                k += 1
                if int(label1) != int(label2):
                    break

        img1 = Image.open(img1)
        img1 = np.array(img1)
        img1 = torch.Tensor(img1)
        img1 = transform(img1)

        img2 = Image.open(img2)
        img2 = np.array(img2)
        img2 = torch.Tensor(img2)
        img2 = transform(img2)

        label = torch.Tensor(np.array([int(label1 != label2)], dtype=np.float32))
        return img1, img2, label

    def __len__(self):
        return len(self.fn_labels)


fn_labels = getFn_Dir(dir_path)
faceDataset = FaceDataset(fn_labels)
train_loader = DataLoader(faceDataset, batch_size=8, shuffle=True)



# =====================================================
vgg19 = models.vgg19(pretrained=True)
vgg19_cnn = vgg19.features
for param in vgg19_cnn.parameters():
    param.requires_grad = False
    # print(param.requires_grad,param.shape)


# 定义孪生网络
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            ......  # 【此处为本例核心代码，已省略，完整代码见教材P149；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            ......  # 【此处为本例核心代码，已省略，完整代码见教材P150；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    def forward_once(self, x):  # torch.Size([8, 1, 100, 100])
        o = x
        o = self.cnn(o)
        o = o.reshape(x.size(0), -1)  # torch.Size([8, 4608])
        o = self.fc1(o)  # torch.Size([8, 512])
        return o

    def forward(self, i1, i2):  # torch.Size([8, 1, 100, 100])

        o1 = self.forward_once(i1)  # 两张图片分别经过卷积网络和全连接网络后：torch.Size([8, 5]) torch.Size([8, 5])
        o2 = self.forward_once(i2)

        return o1, o2


# 定义损失函数类
class LossFunction(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(LossFunction, self).__init__()
        self.margin = margin

    def forward(self, i1, i2, y):  # torch.Size([8, 5]) ,torch.Size([8, 1])
        # 欧氏距离：两个数字相减，平方，sqrt，没有除以n。形状必须是两个维度以上
        dist = torch.pairwise_distance(i1, i2, keepdim=True)  # torch.Size([8, 1])
        loss = torch.mean((1 - y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss


# ------------------------------------
siameseNet = SiameseNet().to(device)
optimizer = optim.Adam(siameseNet.parameters(), lr=0.001)  # ,lr = 0.0005
lossFunction = LossFunction()

start=time.time() 		#开始计时

for ep in range(100):
    for i,(b_img1,b_img2, b_label) in enumerate(train_loader):
        #torch.Size([8, 1, 100, 100]) torch.Size([8, 1, 100, 100]) torch.Size([8])
        b_img1, b_img2, b_label = b_img1.to(device),b_img2.to(device), b_label.to(device)


        pre_o1, pre_o2 = siameseNet(b_img1, b_img2) #torch.Size([8, 5]) torch.Size([8, 5])
        loss = lossFunction(pre_o1, pre_o2, b_label)
        if i%50==0:
            print(ep,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''

'''
end=time.time() #计时结束
print('训练耗时：',round((end-start)/60.0,1),'分钟')
#训练结束

#torch.save(siameseNet,'siameseNet')  #保存模型
siameseNet = torch.load('siameseNet') #加载训练的模型





# ============= 以下开始测试 ===========================
def getImg(fn):
    img = Image.open(fn)
    img2 = img.convert('RGB')  # 用于显示
    img = np.array(img)
    img = torch.Tensor(img)
    img = transform(img)
    return img


def getImg_show(fn):
    img = Image.open(fn)
    img = img.convert('RGB')  # 用于显示
    img = np.array(img)
    return img


path = './data/faces/testing'
fn_labels = getFn_Dir(path)
correct = 0
for fn, label in fn_labels:
    img = getImg(fn).unsqueeze(0).to(device)
    img_min, dist_min, label_min, fn_min = -1, 1000, -1, -1
    for fn2, label2 in fn_labels:
        if fn == fn2:
            continue
        img2 = getImg(fn2).unsqueeze(0).to(device)
        pre_o1, pre_o2 = siameseNet(img, img2)
        dist = torch.pairwise_distance(pre_o1, pre_o2, keepdim=True)
        if dist_min > dist.item():
            dist_min = dist.item()
            img_min = img2
            label_min = label2
            fn_min = fn2
    # img,img_min
    correct += int(label == label_min)
    # print(label,label_min)
    img_show = getImg_show(fn)
    img_show2 = getImg_show(fn_min)

    images = dict()
    images[fn] = img_show
    images[fn_min] = img_show2


    def showTwoImages(images, stitle='', rows=1, cols=1):
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        for idx, title in enumerate(images):
            ax.ravel()[idx].imshow(images[title])
            ax.ravel()[idx].set_title(title)
            ax.ravel()[idx].set_axis_off()
        plt.tight_layout()
        plt.suptitle(stitle, fontsize=18, color='red')
        plt.show()


    stitle = 'Similarity: %.2f' % (dist_min)
    #showTwoImages(images, stitle, 1, 2)

print('一共测试了{:.0f}张图片，准确率为{:.1f}%'\
      .format(len(fn_labels), 100. * correct / len(fn_labels)))




'''


训练损失函数值的结果：


96 0.014447139576077461
97 0.01660165563225746
98 0.015442222356796265
99 0.008568278513848782
训练耗时： 4.5 分钟
一共测试了40张图片，准确率为100.0%



'''