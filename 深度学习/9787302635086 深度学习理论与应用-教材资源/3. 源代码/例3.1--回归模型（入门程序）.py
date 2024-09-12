import torch
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(123)
#-------------------------------
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
#张量化
X1,X2,X3,Y = torch.Tensor(X1),torch.Tensor(X2), torch.Tensor(X3), torch.LongTensor(Y)
#按列“组装”特征值，形成特征张量，形状为torch.Size([2040, 3])
X = torch.stack((X1,X2,X3),dim=1)
#等价于下列语句
#X = torch.cat((X1.view(-1,1),X2.view(-1,1),X3.view(-1,1)),dim=1)


del X1,X2,X3
index = torch.randperm(len(Y))    #随机打乱样本的顺序，注意保持X和Y的一致性
X,Y = X[index],Y[index]         #Y的形状为torch.Size([2040])

#定义类Model3_1
class Model3_1(nn.Module):
    def __init__(self):
        super(Model3_1, self).__init__()
        self.fc1 = nn.Linear(3, 4)  #用于构建神经网络的第1层（隐含层），
                                    #其中包含4个神经元
        self.fc2 = nn.Linear(4, 2)  #用于构建神经网络的第2层（输出层），
                                    #其中包含2个神经元，因为有2个类别
    def forward(self,x):     #实现网络的逻辑结构
        out = self.fc1(x)
        out = torch.tanh(out) 	  #该激活函数可用可不用
        out = self.fc2(out)
        #此处不宜用激活函数sigmoid，因为下面的损失函数会用到
        return out
#----------------------------------------------------------
model3_1 = Model3_1()
optimizer = torch.optim.Adam(model3_1.parameters(), lr=0.01)
LS = []
for epoch in range(5):
    i = 0
    for x,y in zip(X,Y):
        #增加在x的第一个维上插入一个长度为1的维，这个是
        #nn.CrossEntropyLoss()需要。这个1可理解为由1个样本构成的数据批量
        x = x.unsqueeze(0)
        pre_y = model3_1(x)  #默认调用forward()方法
        #nn.CrossEntropyLoss()函数要求y的数据类型为long
        y = torch.LongTensor([y])
        loss = nn.CrossEntropyLoss()(pre_y,y)  #交叉熵损失函数
        #print(loss.item())
        if i%50==0:
            LS.append(loss.item())  #采样损失函数值，用于画图
        i += 1
        optimizer.zero_grad()  	#梯度清零
        loss.backward()      	#反向计算梯度
        optimizer.step()    	#参数更新
#绘制损失函数值的变化趋势
plt.plot(LS)
plt.tick_params(labelsize=13)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.grid()
plt.xlabel("损失函数值采样次序",fontsize=13)
plt.ylabel("交叉熵损失函数值",fontsize=13)
plt.show()
