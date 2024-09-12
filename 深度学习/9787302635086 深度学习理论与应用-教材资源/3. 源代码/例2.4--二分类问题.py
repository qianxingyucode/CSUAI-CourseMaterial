import torch
import matplotlib.pyplot as plt
#读入数据
X1=[2.49, 0.50, 2.73, 3.47, 1.38, 1.03, 0.59, 2.25, 0.15, 2.73]
X2=[2.86, 0.21, 2.91, 2.34, 0.37, 0.27, 1.73, 3.75, 1.45, 3.42]
Y = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1] 	#类标记
X1 = torch.Tensor(X1)
X2 = torch.Tensor(X2)
X = torch.stack((X1,X2),dim=1)  #将所有特征数据“组装”为一个张量
#形状为torch.Size([10, 2])
Y = torch.Tensor(Y) 			#形状为torch.Size([10])
rate = torch.Tensor([0.1]) 		#设置学习率
class Perceptron2():
    def __init__(self):
        self.w1 = torch.Tensor([0.0])  #定义三个待优化参数（属性）
        self.w2 = torch.Tensor([0.0])
        self.b = torch.Tensor([0.0])

    def f(self, x):    			#感知器函数的实现代码
        x1, x2 = x[0], x[1]
        t = self.w1 * x1 + self.w2 * x2 + self.b
        z = 1.0 / (1 + torch.exp(t)) 	#运用sigmoid函数
        return z
    def forward_compute(self, x):  	#前向计算
        pre_y = self.f(x)
        return pre_y
#---------------------------------------------
perceptron2 = Perceptron2() 			#创建实例perceptron2
for ep in range(100):  				#迭代代数为100
    for (x, y) in zip(X, Y):
        pre_y = perceptron2.forward_compute(x)  #执行前向计算
        x1, x2 = x[0], x[1]
        ......  # 【此处为本例核心代码，已省略，完整代码见教材P45；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        perceptron2.w1 = perceptron2.w1 + rate * dw1  #更新w1
        perceptron2.w2 = perceptron2.w2 + rate * dw2  #更新w2
        perceptron2.b = perceptron2.b + rate * db     #更新b
s = '学习到的感知器：pre_y = sigmoid(%0.2f*x1 + %0.2f*x2 + %0.2f)'\
%(perceptron2.w1,perceptron2.w2,perceptron2.b)
print(s)
for (x, y) in zip(X, Y):  #测试结果
    t = 1 if perceptron2.f(x) > 0.5 else 0
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
