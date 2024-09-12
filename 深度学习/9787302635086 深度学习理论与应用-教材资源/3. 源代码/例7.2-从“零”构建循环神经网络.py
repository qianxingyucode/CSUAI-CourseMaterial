import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
seq_len = 4  #序列长度
vec_dim = 12  #定义表示序列中每个元素的向量的长度
data = read_csv(r'./data/international-airline-passengers.csv', usecols=[1], \
engine='python', skipfooter=0)
data = np.array(data)  			#(144, 1)
sc = MinMaxScaler()
data = sc.fit_transform(data)  	#归一化
data = data.reshape(-1, vec_dim)	#torch.Size([12, 12])
train_x,train_y = [],[]
for i in range(data.shape[0] - seq_len): 	#构造8个长度为4的子序列及其后的值
    tmp_x = data[i:i + seq_len, :]		#子序列
    tmp_y = data[i + seq_len, :]		#子序列后面的值
    train_x.append(tmp_x)
    train_y.append(tmp_y)
train_x = torch.FloatTensor(train_x) 	#张量化
train_y = torch.FloatTensor(train_y)
#定义处理序列的类

class My_RNN(nn.Module):
    def __init__(self,n=vec_dim,s=128,m=vec_dim):
        super(My_RNN, self).__init__()
        self.s = s
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P191；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    def forward(self, x):
        a_t_1 = torch.rand(x.size(0), self.s)
        lp = x.size(1)
        for k in range(lp):
            input1 = x[:, k, :]
            input1 = self.U(input1)
            input2 = self.W(a_t_1)
            input = input1 + input2
            a_t = torch.relu(input)
            a_t_1 = a_t
        y_t = self.V(a_t)
        return y_t

#----------------------------------------
air_Model = My_RNN()
optimizer = torch.optim.Adam(air_Model.parameters(), lr=0.01)
#开始训练：
for ep in range(400):
    for i, (x,y) in enumerate(zip(train_x,train_y)):
        x = x.unsqueeze(0)  #改变形状为torch.Size([1, 4, 12])，1为批中的样本数
        pre_y = air_Model(x)  			#torch.Size([1, 12])
        pre_y = torch.squeeze(pre_y) 	#torch.Size([12])
        loss = torch.nn.MSELoss()(pre_y, y)	#计算损失函数值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()	#更新梯度
        if ep % 50 == 0:
            print('epoch:{:3d}, loss:{:6.4f}'.format(ep, loss.item()))
#训练完毕
torch.save(air_Model,'air_Model')
air_Model = torch.load('air_Model')
air_Model.eval()
pre_data = []	#保存预测数据
for i, (x,y) in enumerate(zip(train_x,train_y)): #产生预测数据
    x = x.unsqueeze(0)
    pre_y = air_Model(x)
    pre_data.append(pre_y.data.numpy())
#------------------------
#以下绘制曲线图，将模型的输出和原始数据绘制在一个坐标系上
plt.figure()
pre_data = np.array(pre_data)
pre_data = pre_data.reshape(-1, 1).squeeze()
x_tick = np.arange(len(pre_data)) + (seq_len * vec_dim) #从seq_len * vec_dim开始
plt.plot(list(x_tick), pre_data, linewidth=2.5, label='预测数据')
ori_data = data.reshape(-1, 1).squeeze()  #(144,)
plt.plot(range(len(ori_data)), ori_data, linewidth=2.5,label='原始数据' )
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel("数据的大小（已归一化）",fontsize=14) #Y轴标签
plt.xlabel("月份的序号",fontsize=14) #Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签simhei
plt.grid()
plt.show()
