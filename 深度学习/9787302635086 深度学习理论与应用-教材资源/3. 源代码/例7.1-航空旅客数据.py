import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
#-----------------------------------------------------
seq_len = 4  # 序列长度(每个序列有4个元素，1个元素是一年，12个月，即12个数据表示一个向量，构成一个元素)
vec_dim = 12  # 序列中每个元素的特征数目。本程序采用的序列元素为一年的旅客，一年12个月，即12维特征。

data = read_csv(r'./data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=0)
data = np.array(data)  #(144, 1)
data2 = data[:,0]
sc = MinMaxScaler()
data = sc.fit_transform(data)  # 归一化
data = data.reshape(-1, vec_dim)  # torch.Size([12, 12])
train_x,train_y = [],[]
......  # 【此处为本例部分核心代码，已省略，完整代码见教材P185；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

train_x = torch.FloatTensor(train_x) #torch.Size([8, 4, 12]) torch.Size([8, 12])
train_y = torch.FloatTensor(train_y)
#------------------------------------------------------
class Air_Model(nn.Module):
    def __init__(self):
        super(Air_Model, self).__init__()
        #        输入x的维度12     隐含层h维度10（神经元个数）   隐含层的层数1 ?  默认batch_first=False,即batch在X的第2个维度
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P180；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        self.linear = nn.Linear(10, vec_dim)
    def forward(self, x): #torch.Size([1, 4, 12])
        _, (h_out, _) = self.lstm(x)  # h_out是序列最后一个元素的hidden state
        h_out = h_out.view(x.shape[0],-1)  # h_out's shape torch.Size([1, 10]) = (n_layer * n_direction, batchsize * hidden_dim), i.e. (1, 10)
        o = self.linear(h_out)
        return o


air_Model = Air_Model()
optimizer = torch.optim.Adam(air_Model.parameters(), lr=0.01)


for ep in range(400):
    for i, (x,y) in enumerate(zip(train_x,train_y)):
        x = x.unsqueeze(0) #加上批   torch.Size([1, 4, 12])
        pre_y = air_Model(x)  #torch.Size([1, 12])
        pre_y = torch.squeeze(pre_y) #torch.Size([12])
        loss = torch.nn.MSELoss()(pre_y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            print('epoch:{:3d}, loss:{:6.4f}'.format(ep, loss.item()))


#---------------------------------------

torch.save(air_Model,'air_Model')

'''
'''
air_Model = torch.load('air_Model')
air_Model.eval()
pre_data = []
for i, (x,y) in enumerate(zip(train_x,train_y)):
    x = x.unsqueeze(0) #加上批   torch.Size([1, 4, 12])
    pre_y = air_Model(x)  #torch.Size([1, 12])
    pre_data.append(pre_y.data.numpy())
    #print(pre_y.data.numpy())

#------------------------


plt.figure()
pre_data = np.array(pre_data)  #(8, 1, 12)
pre_data = pre_data.reshape(-1, 1).squeeze() #(8, 12) ---> (96,)

x_tick = np.arange(len(pre_data)) + (seq_len * vec_dim)
plt.plot(list(x_tick), pre_data, linewidth=2.5,   label='预测数据')  #从48开始
#------
ori_data = data.reshape(-1, 1).squeeze()  #(144,)

plt.plot(range(len(ori_data)), ori_data, linewidth=2.5,label='原始数据' ) #  据'

#plt.rcParams['font.sans-serif']=['SimHei']
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel("数据的大小（已归一化）",fontsize=14) #Y轴标签

plt.xlabel("月份的序号",fontsize=14) #Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签simhei
plt.grid()
plt.show()


exit(0)
'''
'''
#绘制原始数据的曲线图=============================
plt.figure()


#------
ori_data = data.reshape(-1, 1).squeeze()  #(144,)

plt.plot(range(len(data2)), data2, linewidth=2.5  ) #  据'

#plt.rcParams['font.sans-serif']=['SimHei']

plt.tick_params(labelsize=14)
plt.ylabel("数据的大小",fontsize=14) #Y轴标签

plt.xlabel("月份的序号",fontsize=14) #Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签simhei
plt.grid()
plt.show()
