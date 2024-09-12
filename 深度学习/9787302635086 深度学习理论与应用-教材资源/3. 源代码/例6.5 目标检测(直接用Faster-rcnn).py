import torch
import torchvision
import numpy as np
import cv2
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def getcolor():  #随机产生颜色
  b = random.randint(0, 255)
  g = random.randint(0, 255)
  r = random.randint(0, 255)
  return (b, g, r)

#类别的名称
names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}




mobj_model = ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P166；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

mobj_model = mobj_model.to(device)
mobj_model.eval()

fn = r'data\object_detection\eating.jpg'
img = cv2.imread(fn) #加载图像
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = torch.Tensor(img2 / 255.).permute(2, 0, 1).to(device) #归一化
imgs = [img2]


# outs是list类型，imgs中有多少张图片，out中就有多少个对象，每个对象都是一个字典  键和值。
outs = mobj_model(imgs)

out = outs[0]  #一个out包含3个元素，元素的键分别为'boxes'、'labels'、'scores'，它们的长度都相同，长度为多少就表示检测到多少个目标
boxes = out['boxes'].data.cpu()
labels = out['labels'].data.cpu()
scores = out['scores'].data.cpu()


for i in range(len(boxes)):
    score = scores[i].item()
    if score < 0.8:
      continue
    class_index = str(labels[i].item())  	#获得下标编号
    class_name = names[class_index]  	#获得类别名称
    box = np.array(boxes[i])
    #框住目标（“框”的坐标参数在box中）
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=getcolor(), thickness=2)
    cv2.putText(img, text=class_name, org=(int(box[0]), 	#在框上方添加类别名称
    int(box[1]) - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
thickness=1,  lineType=cv2.LINE_AA, color=(0, 0, 255))
img = np.array(img)
cv2.imshow('11',img)  #显示效果
cv2.waitKey(0)


