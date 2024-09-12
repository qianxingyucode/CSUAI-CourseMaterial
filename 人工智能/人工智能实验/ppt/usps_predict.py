import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
import random

BATCH_SIZE=32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transforms=transforms.Compose([
                           transforms.Grayscale(num_output_channels=1),
                           transforms.Resize((28, 28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

Transforms1=transforms.Compose([
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        in_size = x.size(0)
        # batch的size
        out = self.conv1(x) #24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out

model = ConvNet().to(DEVICE)

file_path = r"./"
te = "t10k-images"
test_data = datasets.ImageFolder(os.path.join(file_path, te), Transforms)

from torch.utils import data

batch_size = 32
test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

def inv(data):
    invdata = torch.Tensor(1, 28, 28)
    p = 0.02
    for i in range(0, 28):
        for j in range(0, 28):
            invdata[0][i][j] = data[0][i][j]
            invdata[0][i][j] *= 0.3081
            invdata[0][i][j] += 0.1307
            rnd = random.random()
            if rnd < p:
                invdata[0][i][j] = 0
            elif rnd > 1 - p:
                invdata[0][i][j] = 1

    return invdata


model.load_state_dict(torch.load('./1.pth'))
model.eval()

plt.Figure()
toPIL = transforms.ToPILImage()
print("预测值为: ", end="")
for data, target in test_loader:
    for i in range(1, 7):
        picdata = inv(data[i - 1])
        pic = toPIL(picdata)
        plt.subplot(2, 3, i)
        plt.imshow(pic, interpolation="None", cmap="gray")
        picdata = Transforms(pic)

        with torch.no_grad():
            picdata = picdata.to(DEVICE)
            output = model(picdata)
            pred = torch.max(output, 1)[1].item()
            print(pred, end=" ")
    break
plt.show()




# def Test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
#             pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# Test(model, DEVICE, test_loader)