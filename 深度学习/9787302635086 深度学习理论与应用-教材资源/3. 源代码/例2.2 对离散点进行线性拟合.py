import torch
import matplotlib.pyplot as plt
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 		#读取数据
Y = [-9.51, -5.74, -2.84, -1.8, 0.54, 1.51, 4.33, 7.06, 9.34, 10.72]
X = torch.Tensor(X)  	#转换为张量
Y = torch.Tensor(Y)
def f(x):  			#定义感知器函数
    t = w*x + b
    return t

w, b = torch.rand(1), torch.rand(1) 	#随机初始化w和b

def dw(x,y):   #目标函数关于w的导数函数：
    t = (f(x) -y) * x
    return t
def db(x,y):   #目标函数关于b的导数函数：
    t = (f(x) -y)
    return t


lr = torch.Tensor([0.01])          #设置学习率
for epoch in range(1000):  #设置循环的代数
    for x,y in zip(X, Y):   #注意，X和Y中的元素一一对应
        dw_v, db_v = dw(x,y), db(x,y)
        w = w - lr * dw_v
        b = b - lr * db_v



plt.scatter(X,Y,c='r')
X2 = [X[0],X[len(X)-1]]   		#过两点绘制感知器函数直线图
Y2 = [f(X[0]),f(X[len(X)-1])]
plt.plot(X2,Y2,'--',c='b')
plt.tick_params(labelsize=13)
plt.show()


