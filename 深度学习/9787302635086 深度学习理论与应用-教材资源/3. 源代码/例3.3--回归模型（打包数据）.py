

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data  import DataLoader,TensorDataset
#读取文件"例3.1数据.txt"中的数据：
path = r'.\\data'
fg=open(path+'\\'+"例3.1数据.txt","r",encoding='utf-8')
s=list(fg)
X1,X2,X3,Y = [],[],[],[]
for i,v in enumerate(s):
    v = v.replace('\n','')
    v = v.split(',')
    X1.append(float(v[0]))
    X2.append(float(v[1]))
    X3.append(float(v[2]))
    Y.append(int(v[3]))
fg.close()
X1,X2,X3,Y = torch.Tensor(X1),torch.Tensor(X2),torch.Tensor(X3),torch.LongTensor(Y)
X = torch.stack((X1,X2,X3),dim=1)
del X1,X2,X3
index = torch.randperm(len(Y))  #随机打乱样本的顺序，注意保持X和Y的一致性
X,Y = X[index],Y[index]       #Y的形状为torch.Size([2040])
#数据集分割：
......  # 【此处为本例核心代码，已省略，完整代码见教材P75；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

batch_size = 100 #设置包的大小（规模）
#对训练集打包：
train_set = TensorDataset(trainX,trainY)
train_loader = DataLoader(dataset=train_set,   #打包
                          batch_size=batch_size, 	#设置包的大小
                          shuffle=False) #默认：shuffle=False
#对测试集打包：
test_set = TensorDataset(testX,testY)
test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,		#设置包的大小
                         shuffle=False) 	#默认shuffle=False
del train_set,test_set
del X,Y
#定义类Model3_1
class Model3_1(nn.Module):
    def __init__(self ):
        super(Model3_1, self).__init__()
        self.fc1 = nn.Linear(3, 4)  #用于构建神经网络的第1层（隐含层）
        self.fc2 = nn.Linear(4, 2)  #用于构建神经网络的第2层（输出层）
    def forward(self,x):
        out = self.fc1(x)
        out = torch.tanh(out)   #增加了一个激活函数
        out = self.fc2(out)
        return out
#----------------------------------------------------------
model3_1 = Model3_1()  #实例化模型
optimizer = torch.optim.Adam(model3_1.parameters(), lr=0.01) #设置优化器
#以下为训练代码：
for epoch in range(10):  #训练代数设置为10
    for x, y in train_loader: #使用上面打包的训练集进行训练
        pre_y = model3_1(x)
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P75；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        optimizer.zero_grad()  	#梯度清零
        loss.backward()       	#反向计算梯度
        optimizer.step()       	#参数更新
#以下开始模型测试，计算预测的准确率：
model3_1.eval()  #设置为测试模式
correct = 0
with torch.no_grad():  #torch.no_grad()是一个上下文管理器，在其中放弃梯度计算
    for x,y in test_loader:
        pre_y = model3_1(x)
        pre_y_index = torch.argmax(pre_y, dim=1) # 找到概率最大的下标
        t = (pre_y_index==y).long().sum()
        correct += t
s = '在测试集上的预测准确率为：{:.1f}%'.format(100.*correct/len(test_loader.dataset))
print(s)
