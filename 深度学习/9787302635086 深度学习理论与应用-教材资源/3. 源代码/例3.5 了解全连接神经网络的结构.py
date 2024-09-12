import torch
import torch.nn as nn
class FcNet(nn.Module):     #定义深度神经网络模型类
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 3, bias=True)    #定义全连接层
        self.layer2 = nn.Linear(3, 2, bias=True)    #定义全连接层
    def forward(self,x):        #“连接”两个全连接层
        o = self.layer1(x)
        #o = torch.sigmoid(o)  #激活函数
        o = self.layer2(o)
        #o = torch.sigmoid(o)  #激活函数
        return o
fcNet = FcNet()             #实例化网络类
W1 = torch.tensor([[2, 1, 3],    #构造权重矩阵W1
                [2, 4, 1],
                [2, 6, 5]])
W2 = torch.tensor([[2, 1, 4],    #构造权重矩阵W2
                [0, 3, 3]])
b1 = torch.tensor([10, 20, -10])  #构造偏置项向量b1
b2 = torch.tensor([20, -30])              #构造偏置项向量b2
......  # 【此处为本例核心代码，已省略，完整代码见教材P85；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

X = torch.tensor([30, 50, 20]).float() 	#构造输入张量X
pre_Y = fcNet(X)       #将X送入模型fcNet进行计算，结果放在张量pre_Y中
print(X.long())
print(pre_Y.data.long())  #输出结果


