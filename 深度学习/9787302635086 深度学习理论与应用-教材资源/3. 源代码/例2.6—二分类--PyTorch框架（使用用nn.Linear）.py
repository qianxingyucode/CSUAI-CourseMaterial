import torch
import matplotlib.pyplot as plt
import torch.nn as nn
torch.manual_seed(123)
#读入数据
X1=[2.49, 0.50, 2.73, 3.47, 1.38, 1.03, 0.59, 2.25, 0.15, 2.73]
X2=[2.86, 0.21, 2.91, 2.34, 0.37, 0.27, 1.73, 3.75, 1.45, 3.42]
Y = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1] 	#类标记
X1 = torch.Tensor(X1)
X2 = torch.Tensor(X2)
X = torch.stack((X1,X2),dim=1)  #将所有特征数据“组装”为一个张量
                          	    #形状为torch.Size([10, 2])
Y = torch.Tensor(Y) 			#形状为torch.Size([10])

#定义类Perceptron2：
class Perceptron2(nn.Module): #必须继承类nn.Module
    def __init__(self):
        super(Perceptron2, self).__init__() #必须添加该方法，第一个参数必须是类名
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P51；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    def forward(self, x): #编写前向计算逻辑
        out = self.fc(x)  #调用感知器，执行前向计算
        out = torch.sigmoid(out) #运用sigmoid激活函数
        return out
#---------------------------------------------
perceptron2 = Perceptron2() 			#创建实例perceptron2
optimizer = torch.optim.Adam(perceptron2.parameters(), lr=0.1) #设计优化器
#以下开始训练
for ep in range(100):  				#迭代代数为100
    for (x, y) in zip(X, Y):
        pre_y = perceptron2(x)  #x的形状为torch.Size([2])
        y = torch.Tensor([y]) 	#为了适合于函数nn.BCELoss()的计算形状
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P51；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        optimizer.zero_grad()	#对参数的梯度清零，即去掉以前保存的梯度
        loss.backward()   		#反向转播并计算各参数的梯度
        optimizer.step()    	#利用梯度更新参数
#至此，训练完毕
t = list(perceptron2.parameters()) #读出触发器中的参数w1、w2和b
w1 = t[0].data[0,0]
w2 = t[0].data[0,1]
b = t[1].data
s = '学习到的感知器：pre_y = sigmoid(%0.2f*x1 + %0.2f*x2 + %0.2f)'\
%(w1,w2,b)
print(s)
perceptron2.eval()  #设置为测试模式


for (x, y) in zip(X, Y):  #使用感知器做预测测试
    t = 1 if perceptron2(x) > 0.5 else 0 #阶跃变换
    s = ''
    if t == y.item():
        s = '点(%0.2f, %0.2f)被<正确>分类！'%(x[0],x[1])
    else:
        s = '点(%0.2f, %0.2f)被<错误>分类！' % (x[0], x[1])
    print(s)


#----- 以下为非必要代码，仅用于绘制散点图和学习到的直线 -----
#绘制散点图
t1 = [i for (i, e) in enumerate(Y) if e == 0]  #获得0类标记在Y中的下标值
t2 = [i for (i, e) in enumerate(Y) if e == 1]  #获得1类标记在Y中的下标值
X1,X2 = X[t1],X[t2]
plt.scatter(X1[:, 0], X1[:, 1],  marker='o',c='k') #在坐标系中绘制0类样本数据点
plt.scatter(X2[:, 0], X2[:, 1],  marker='v',c='k') #在坐标系中绘制1类样本数据点
#绘制直线
def g(x1):
    x2 = -(w1 * x1 + b) / w2
    return x2
xmin,xmax = X[:,0].min(), X[:,0].max()
T1 = [xmin, xmax]
T2 = [g(xmin),g(xmax)]
plt.plot(T1, T2, '--', c='k')
plt.grid()
plt.tick_params(labelsize=13)
plt.xlabel("x$_1$",fontsize=13)
plt.ylabel("x$_2$",fontsize=13)
plt.show()
