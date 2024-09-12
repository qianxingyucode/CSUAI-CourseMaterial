import torch
import matplotlib.pyplot as plt
import torch.nn as nn
#读入数据
X1=[2.49, 0.50, 2.73, 3.47, 1.38, 1.03, 0.59, 2.25, 0.15, 2.73]
X2=[2.86, 0.21, 2.91, 2.34, 0.37, 0.27, 1.73, 3.75, 1.45, 3.42]
Y = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1] 	#类标记
X1 = torch.Tensor(X1)
X2 = torch.Tensor(X2)
X = torch.stack((X1,X2),dim=1)  #将所有特征数据“组装”为一个张量
                          	#形状为torch.Size([10, 2])
Y = torch.Tensor(Y) 			#形状为torch.Size([10])

#定义类Perceptron2
class Perceptron2(nn.Module): #必须继承类nn.Module
    def __init__(self):
        super().__init__() #添加该方法，第一个参数是类名，或者去掉所有的参数
        # 定义三个待优化参数（属性）
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P49；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    def f(self, x):    				#感知器函数的实现代码
        x1, x2 = x[0], x[1]
        t = self.w1 * x1 + self.w2 * x2 + self.b
        z = 1.0 / (1 + torch.exp(t))	#运用sigmoid函数
        return z
    def forward(self, x): #该方法的名称是固定的，用于编写前向计算逻辑
        pre_y = self.f(x)
        return pre_y
#---------------------------------------------
perceptron2 = Perceptron2() 			#创建实例perceptron2
optimizer = torch.optim.Adam(perceptron2.parameters(), lr=0.1) #设计优化器
#以下开始训练
for ep in range(100):  				#迭代代数为100
    for (x, y) in zip(X, Y):
        pre_y = perceptron2(x)  	#执行前向计算，多用perceptron2(x)形式，
#其等效于perceptron2.forward(x)
        y = torch.Tensor([y]) 		#为了适合于函数nn.BCELoss()的计算，
#将y的形状由torch.Size([])改为torch.Size([1])
        loss = nn.BCELoss()(pre_y, y)  # nn.Module提供的目标函数


        optimizer.zero_grad()	#对参数的梯度清零，去掉以前保存的梯度
        loss.backward()   		#反向转播并计算各参数的梯度
        optimizer.step()    	#执行利用梯度更新参数
#至此，训练完毕
s = '学习到的感知器：pre_y = sigmoid(%0.2f*x1 + %0.2f*x2 + %0.2f)'\
%(perceptron2.w1,perceptron2.w2,perceptron2.b)

print(s)
for (x, y) in zip(X, Y):  #使用感知器做预测测试
    t = 1 if perceptron2.f(x) > 0.5 else 0 #阶跃变换
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
plt.scatter(X1[:, 0], X1[:, 1],  marker='o',c='r') #在坐标系中绘制0类样本数据点
plt.scatter(X2[:, 0], X2[:, 1],  marker='v',c='g') #在坐标系中绘制1类样本数据点
#绘制直线
def g(x1):
    x2 = -(perceptron2.w1 * x1 + perceptron2.b) / perceptron2.w2
    return x2
xmin,xmax = X[:,0].min(), X[:,0].max()
T1 = [xmin, xmax]
T2 = [g(xmin),g(xmax)]
plt.plot(T1, T2, '--', c='b')
plt.grid()
plt.tick_params(labelsize=13)
plt.xlabel("x$_1$",fontsize=13)
plt.ylabel("x$_2$",fontsize=13)
plt.show()
