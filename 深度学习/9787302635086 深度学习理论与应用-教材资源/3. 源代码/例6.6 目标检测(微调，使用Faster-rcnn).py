import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import os
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
import random
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#定义数据集类
class GetDataset(Dataset):
    def __init__(self, dir_path):
        #获取所有文件的路径和文件名
        self.all_images = [os.path.join(dir_path,file) for file in os.listdir(dir_path) if '.jpg' in file]
    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        img = cv2.imread(img_name)  #读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_resized = cv2.resize(img, (600, 600)) #调整尺寸
        img_resized /= 255.0  #归一化
        # 获得对应的xml文件，读取标注信息
        annot_fn = img_name[:-4] + '.xml'  #图像名的后缀为.jpg,而标注的xml文件的后缀为.xml，而二者的主文件名相同
        boxes,labels = [],[]
        img_width = img.shape[1]
        img_height = img.shape[0]
        tree = et.parse(annot_fn)
        root = tree.getroot()
        #一个人脸使用一个框架（box）来框住（定位），因此这个人脸的左上角坐标和右下角坐标即可唯一定位这个人脸
        #xml文件中包含了多个人脸的box坐标参数（四个参数），下列代码用于提取xml文件中描述的所有box的坐标参数
        for member in root.findall('object'):  # 把多个人脸的框架坐标保存下来
            label = 1 if member.find('name').text == 'face' else 0 #1表示人脸，0表示不是
            labels.append(label)
            x1 = int(member.find('bndbox').find('xmin').text) #左上x坐标
            y1 = int(member.find('bndbox').find('ymin').text) #左上y坐标
            x2 = int(member.find('bndbox').find('xmax').text) #右下x坐标
            y2 = int(member.find('bndbox').find('ymax').text) #右下y坐标
            ##根据所需的width，height，按比例缩放边界框的大小
            xx1 = int((x1 / img_width) * 600)
            yy1 = int((y1 / img_height) * 600)
            xx2 = int((x2 / img_width) * 600)
            yy2 = int((y2 / img_height) * 600)
            boxes.append([xx1, yy1, xx2, yy2])
        boxes = torch.LongTensor(boxes)  # 张量化
        labels = torch.LongTensor(labels)
        target = {}  # 长度为2
        target["boxes"] = boxes  #有多少个对象，就有多少个box
        target["labels"] = labels  #每一个box（对象——人脸）有一个类编号
        T = transforms.Compose([transforms.ToTensor()])
        img_resized = T(img_resized)
        return img_resized, target
    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    return tuple(zip(*batch))  #按列组装batch中的元素，这样列变成行，相当于对batch中的数据矩阵转置了
#--------------------------------------
train_dataset = GetDataset('./data/face_detection/train')
test_dataset = GetDataset('./data/face_detection/valid')
train_loader = DataLoader(train_dataset, batch_size=16,
shuffle=True,num_workers=0, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True,
num_workers=0, collate_fn=collate_fn)

print(f"训练集样本数: {len(train_loader.dataset)}") #9146
print(f"测试集样本数: {len(test_loader.dataset)}")


#-----------------------------------
# 加载预训练模型以
face_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                  trainable_backbone_layers=0)
......  # 【此处为本例部分核心代码，已省略，完整代码见教材P169；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

#--- 以下开始训练模型 -----
face_model.train()
for epoch in range(5):
    for i, (imgs, targets) in enumerate(train_loader):
        try:
            imgs = list(img.to(device) for img in imgs) #有16张图片在里面
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #每张图片有一个字典
            pre_dict = face_model(imgs, targets)
            losses = sum(loss for loss in pre_dict.values())
            print(epoch,losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        except:
            print('标注可能有问题.........')
torch.save(face_model,'face_model')
'''
'''
#--- 以下开始测试模型 -----
face_model = torch.load('face_model')
face_model.eval()

'''
with torch.no_grad():
    for i, (imgs, targets) in enumerate(test_loader):
        imgs = list(img.to(device) for img in imgs) #img: torch.Size([3, 600, 600])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        pre_dict = face_model(imgs)  #在训练模式下不用targets,pre_dict为长度为16（批大小）的list，里面的元素是字典
        index = 0 #选择其中的一张照片来展示人脸检测的效果

        adict = pre_dict[index]  #字典的长度为3，#关键字为：boxes,labels,scores
        img = imgs[index].permute([1, 2, 0]).data.cpu()
        img = np.array(img)


        for k, box in enumerate(adict['boxes']):
            score = adict['scores'][k]
            if score<0.6:
                continue
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                          color=(0, 0, 255), thickness=2)
            #还可以统计box的重叠区域来计算准确率等，在此略过........

        cv2.imshow('Object detection', img)
        cv2.waitKey(0)
'''


'''
'''
#-------- 检测单张照片中的人脸 ------------------
img  = cv2.imread('./data/object_detection/eating.jpg')  # 读取图像
#img  = cv2.imread(r'data\face_detection\test\21_Festival_Festival_21_106.jpg')
img = cv2.resize(img, (600, 600)) 	#调整尺寸
img2 = img
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img /= 255.0  	#归一化
T = transforms.Compose([transforms.ToTensor()])
img = T(img).to(device)
imgs = [img]
pre_dict = face_model(imgs)
adict = pre_dict[0]
for k, box in enumerate(adict['boxes']):
    score = adict['scores'][k]
    if score < 0.6:
        continue
    cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]),
                  color=(0, 0, 255), thickness=2)
cv2.imshow('Face detection', img2)
cv2.waitKey(0)

