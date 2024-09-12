import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = models.vgg16_bn(pretrained=True)  #加载VGG16
model.eval()
def hook_fun(model, input, output):     		#定义前向hook
    global out_FM   #定义全局变量，用于存放输出的特征图
    out_FM = output
    return None
model.avgpool.register_forward_hook(hook_fun) #注册一个前向hook



tfs = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
transforms.ToTensor(),
                      transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])])
path = r'./data/Interpretability/images'
name = 'both.png'


img_path = path + '\\' + name
img = Image.open(img_path).convert('RGB')  #打开图片并转换为RGB模型
origin_img = img  	#保存原图
img = tfs(img)     	#转化为张量
img = torch.unsqueeze(img, 0)     #增加batch维度 torch.Size([1, 3, 224, 224])
#下列语句在前向计算而执行到model.layer4[1].bn2层时会调用到hook_fun()函数
#传入该函数的分别是model.layer4[1].bn2层本身以及该层的输入和输出
out = model(img) #执行该语句后，out_FM才产生  #torch.Size([1, 1000])
pre_y = torch.argmax(out).item() #预测类别



def grad_hook(grad): 	#定义一个后向hook
    ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P279；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    return None

out_FM.register_hook(grad_hook) 	#对out_FM注册一个反向hook

pre_class = out[:, pre_y]
pre_class.backward()  #执行该语句后，temp_grad才产生
#out_FM和temp_grad的形状完全一样——torch.Size([1, 512, 7, 7])，
#但前者是特征图，后者是关于特征图的导数，是构造权重的依据
out_FM = out_FM[0] #torch.Size([512, 7, 7])
temp_grad = temp_grad[0] #torch.Size([512, 7, 7])

weights = torch.nn.AdaptiveAvgPool2d((1, 1))(temp_grad) #对各通道平均池化，torch.Size([512, 1, 1])
#(512, 1, 1)*(512, 7, 7)-->(512, 7, 7)
weighted_FM = weights * out_FM
weighted_FM = weighted_FM.sum(0) #torch.Size([7, 7])

#------------------以下不变----------------------
#运用激活函数,仅保留非负值; 平方的目的是为了降低小特征值的干扰
weighted_FM = torch.relu(weighted_FM)**2
weighted_FM = (weighted_FM-weighted_FM.min()) \
              /(weighted_FM.max()-weighted_FM.min()) #归一化
#将weighted_FM转换成PIL格式，以条用resize()函数
weighted_FM = to_pil_image(np.array(weighted_FM.detach()), mode='F')
#通过插值，将加权求和后的单通道特征图扩充为跟原图一样大小


expanded_FM = weighted_FM.resize(origin_img.size, resample=Image.BICUBIC)

#运用调色板，将expanded_FM转化为可视图，其中值大的像素显示为红色或深蓝
#色，值小的为浅蓝色
expanded_FM = 255 * cm.get_cmap('jet')(np.array(expanded_FM))[:, :, 1:]
expanded_FM = expanded_FM.astype(np.uint8)
#将原图和可视化后的单通道特征图叠加（融合），形成类激活图CAM
CAM = cv2.addWeighted(np.array(origin_img), 0.6, np.array(expanded_FM), 0.4, 0)
plt.imshow(CAM) #显示类激活图CAM
plt.show()



