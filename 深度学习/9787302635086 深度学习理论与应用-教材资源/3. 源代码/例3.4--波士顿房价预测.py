import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data  import DataLoader,TensorDataset
torch.manual_seed(123)
#-------------------------------
#读取文件"housing.data"中的数据：
path = r'.\\data'
fg=open(path+'\\'+"housing.data","r",encoding='utf-8')
s=list(fg)
X,Y = [],[]
for i,line in enumerate(s):
    line = line.replace('\n','')
    line = line.split(' ')
    line2 = [float(v) for v in line if v.strip()!='']
    X.append(line2[:-1]) #取得特征值向量
    Y.append(line2[-1])  #取样本标记（房屋价格）
fg.close()
X = torch.FloatTensor(X) #torch.Size([506, 13]) torch.Size([506])
Y = torch.FloatTensor(Y)

index = torch.randperm(len(X))
X,Y = X[index],Y[index]  #随机打乱顺序


torch.manual_seed(124)


rate = 0.8
train_len = int(len(X)*rate)
trainX,trainY = X[:train_len],Y[:train_len] #训练集
testX,testY = X[train_len:],Y[train_len:]   #测试集

#训练集和测试集一般要分开归一化，但归一化方法应一样：
def map_minmax(T): #归一化函数
    min,max = torch.min(T,dim=0)[0],torch.max(T,dim=0)[0]
    r = (1.0*T-min)/(max-min)
    return r
trainX,trainY = map_minmax(trainX),map_minmax(trainY)
testX,testY = map_minmax(testX),map_minmax(testY)
#--------------------
batch_size = 16 #设置包的大小
#对训练集打包：
train_set = TensorDataset(trainX,trainY)
train_loader = DataLoader(dataset=train_set,   #打包
                          batch_size=batch_size,
                          shuffle=False) #默认：shuffle=False

#对测试集打包：
test_set = TensorDataset(testX,testY)
test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,
                         shuffle=False) #默认：shuffle=False
del X,Y,trainX,trainY,testX,testY,train_set,test_set

#定义类Model2_2
class Model2_2(nn.Module):
    def __init__(self ):
        super(Model2_2, self).__init__()
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P80；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    def forward(self,x):
        out = self.fc1(x)
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P80；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        return out

model2_2 = Model2_2()
optimizer = torch.optim.Adam(model2_2.parameters(), lr=0.01) #lr=0.005

ls = []
for epoch in range(200):
    for i,(x, y) in enumerate(train_loader): # 使用上面打包的训练集进行训练
        pre_y = model2_2(x) #pre_y的形状为torch.Size([30, 1])
        pre_y = pre_y.squeeze() #改为torch.Size([30])

        loss = nn.MSELoss()(pre_y,y) #均方差损失函数
        #print(loss.item())
        if i%100==0:
            ls.append(loss.item())

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向计算梯度
        optimizer.step()  # 参数更新





#以下开始模型测试，计算预测的准确率：
lsy = torch.Tensor([])
ls = torch.Tensor([])
model2_2.eval()  #设置为测试模式
correct = 0
with torch.no_grad():  #torch.no_grad()是一个上下文管理器，在该管理器中放弃梯度计算
    for x,y in test_loader:
        pre_y = model2_2(x)  #torch.Size([16, 1])
        pre_y = pre_y.squeeze()
        t = (torch.abs(pre_y-y)<0.1)
        t = t.long().sum()
        correct += t

        #print(pre_y.shape)
        #X = torch.stack((X1, X2, X3), dim=1)
        ls = torch.cat((ls, pre_y))
        lsy = torch.cat((lsy, y))
        #print(ls.shape)

s = '在测试集上的预测准确率为：{:.1f}%'.format(100.*correct/len(test_loader.dataset))
print(s)

plt.plot(ls,label='实际值')
plt.plot(lsy,label='预测值')
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签simhei
plt.xlabel("采样点序号",fontsize=16) #X轴标签
plt.ylabel("房屋价格（归一化后）",fontsize=16) #Y轴标签
plt.tick_params(labelsize=16)
plt.grid()
plt.legend()

plt.show()

exit(0)

