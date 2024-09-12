import torch
import matplotlib.pyplot as plt
torch.manual_seed(123)
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [-9.51, -5.74, -2.84, -1.8, 0.54, 1.51, 4.33, 7.06, 9.34, 10.72]
X = torch.Tensor(X)  #转换为张量
Y = torch.Tensor(Y)

X8,Y8 = X,Y
#数据打包：
n = 4 #设置包的大小
X1,Y1 = X[0:n],Y[0:n]         #该包的大小为4
X2,Y2 = X[n:2*n],Y[n:2*n]     #该包的大小为4
X3,Y3 = X[2*n:3*n],Y[2*n:3*n] #该包的大小为2

X,Y = [X1,X2,X3],[Y1,Y2,Y3]   #重新定义X和Y



#定义感知器的函数
def f(x):
    t = w*x + b
    return t


def dw(x,y):   #目标函数关于w的导数函数：
    ...... #【此处为本例核心代码，已省略，完整代码见教材P42；建议读者手工输入核心代码并进行调试，这样方能领会其含义】
    return t
def db(x,y):   #目标函数关于b的导数函数：
    t = (f(x) -y)
    return t

#-----------------------------
w,b = torch.rand(1), torch.rand(1)  #随机初始化w和b
rate = torch.Tensor([0.01])          #设置学习率
for epoch in range(1000): #设置循环的代数
    for bX,bY in zip(X,Y):  #此处与例1.2不同
        dw_v, db_v = dw(bX, bY), db(bX, bY)  #由于bX和bY均为张量，所以函数dw和db不需要修改
        dw_v = dw_v.mean() #求平均值
        db_v = db_v.mean()
        w = w - rate * dw_v
        b = b - rate * db_v
print("优化后，参数w和b的值分别为：%0.4f和%0.4f"%(w,b))




X,Y=X8,Y8

plt.scatter(X,Y,c='k')

X2 = [X[0],X[len(X)-1]]  #过两点绘制感知器函数直线图
Y2 = [f(X[0]),f(X[len(X)-1])]
plt.plot(X2,Y2,'--',c='k')
plt.tick_params(labelsize=13)
plt.show()

 