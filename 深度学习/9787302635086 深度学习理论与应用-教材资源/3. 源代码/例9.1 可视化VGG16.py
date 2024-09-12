
import torch
from torchvision import models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm


model = models.vgg16_bn(pretrained=True)  #加载VGG16
model.eval()

cnn_layers = []
for k in range(43+1):  #添加卷积网络层
    cnn_layers.append(model.features[k])
cnn_layers.append(model.avgpool)  #自适应平均池化层
path = r'./data/Interpretability/images'
name = 'both.png'
fn = path + '\\' + name
tfs = transforms.Compose([transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])])
origin_img = Image.open(fn).convert('RGB')    #打开图片并转换为RGB模型
img = tfs(origin_img)     #图片预处理  torch.Size([3, 224, 224])
img = img.unsqueeze(0)   #添加批次的维  torch.Size([1, 3, 224, 224])
def showImg(channel_img):  #channel_img表示特征图的一个通道图像
    channel_img = torch.relu(channel_img)  #torch.Size([224, 224])
    max_min = channel_img.max()-channel_img.min()
    if max_min == 0:
        max_min = 1e-6
        channel_img = (channel_img-channel_img.min())/max_min  #归一化到[0,1]
    channel_img = (channel_img**2)*255  #在归一化到[0, 255]
    img = np.array(channel_img.data).astype(np.uint8)
    cmap = cm.get_cmap('jet')  	#定义调色板
    img = cmap(img )[:, :, 1:]   	#使用调色板
    plt.imshow(img) 			#显示图像
    plt.show()
    return


out = img
for k,m in enumerate(cnn_layers):
    out = m(out)
    if k == 3:  #cnn_layers[3]存放VGG16的第二个卷积层
        print('第二个卷积层输出特征图的形状：', out.shape)
        showImg(out[0, 0])
        showImg(out[0, 20])
        showImg(out[0, 40])
        showImg(out[0, 60])
    elif k == 44:  #cnn_layers[44]存放VGG16的自适应平均池化层
        print('自适应平均池化层输出特征图的形状：', out.shape)
        showImg(out[0, 0])
        showImg(out[0, 20])
        showImg(out[0, 40])
        showImg(out[0, 60])

out = img
for k,m in enumerate(cnn_layers):
    out = m(out)
    if k == 44:
        img44_60 = out[0, 60]
#归一化到[0,1]
img44_60 = (img44_60-img44_60.min())/(img44_60.max()-img44_60.min())
img44_60 = np.array(img44_60.data)  #转化为数组
#转化为PIL格式，以准备调用resize()方法进行插值缩放
img44_60 = to_pil_image(img44_60, mode='F')
h,w,_ = np.array(origin_img).shape  #获取原图的尺寸
#通过双三次插值方法扩展为跟原图一样大小
......  # 【此处为本例部分核心代码，已省略，完整代码见教材P273；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

cmap = cm.get_cmap('jet')
over_img = cmap(over_img)[:, :, 1:]  #使用调色板
over_img = (255*over_img).astype(np.uint8)  #over_img为数组类型
origin_img = np.array(origin_img)
a = 0.7
#融合两张图片
origin_over_img = Image.fromarray((a * origin_img + (1 - a) * over_img).astype(np.uint8))
plt.imshow(over_img ) #显示扩展后的通道图像
plt.show()
plt.imshow(origin_over_img ) #显示融合后的图像
plt.show()








