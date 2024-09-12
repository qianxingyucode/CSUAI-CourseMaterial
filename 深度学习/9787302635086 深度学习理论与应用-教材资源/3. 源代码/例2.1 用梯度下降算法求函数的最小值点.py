import torch
def f(w): #定义函数
    t = 2 * (w - 2) ** 2 + 1
    return t
def df(w): 				#函数的导数
    t = 4 * (w - 2)
    return t
lr = 0.1  #学习率
w = torch.Tensor([5.0]) 	#设置寻找的起点
for epoch in range(20): 		#迭代循环
    w = w - lr*df(w)  	#更新w
y = f(w)
w, y = round(w.item(),2),round(y.item(),2)
print("该函数的最小值点是：(%0.2f,%0.2f)"%(w, y))   #输出最小值点
