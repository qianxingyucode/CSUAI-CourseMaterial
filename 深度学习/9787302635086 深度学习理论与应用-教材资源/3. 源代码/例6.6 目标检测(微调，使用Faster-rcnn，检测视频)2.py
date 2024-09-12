import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
face_model = torch.load('face_model').to(device) #加载训练好的模型
face_model.eval()
def detect_face(img_t):  #定义函数，其作用是：对输入的图像进行人脸检测
    #img_t  = cv2.imread(fn)  				# 读取图像
    img_t = cv2.resize(img_t, (600, 600)) 		#调整尺寸
    img_t2 = img_t
    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_t /= 255.0  	#归一化
    T = transforms.Compose([transforms.ToTensor()])
    img_t = T(img_t).to(device)
    img_ts = [img_t]
    pre_dict = face_model(img_ts)  #调用训练好的模型
    adict = pre_dict[0]
    for k, box in enumerate(adict['boxes']):
        score = adict['scores'][k]
        if score < 0.6:
            continue
        cv2.rectangle(img_t2, (box[0], box[1]), (box[2], box[3]),
                      color=(0, 0, 255), thickness=2)
    return img_t2 #返回已标记的图像
cap = cv2.VideoCapture(0)
while True:
    ret, image_np = cap.read()  	#打开摄像头
    img = cv2.resize(image_np, (600, 600))
    img = detect_face(img)  	#调用函数
    cv2.imshow('Face detection', img )
    if cv2.waitKey(42) == 27: #当按【esc】键退出, 或者强行终止程序运行也可以退出
        break
