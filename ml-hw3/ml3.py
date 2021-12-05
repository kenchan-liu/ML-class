import numpy as np
import matplotlib.pyplot as plt

I=[[2,0],[0,2]]
i1=np.random.multivariate_normal([1,1],I,(600,1))
i2=np.random.multivariate_normal([8,1],I,(100,1))
i3=np.random.multivariate_normal([4,4],I,(300,1))
for i in range(600):
    plt.plot(i1[i][0][0],i1[i][0][1],'r.')
for i in range(100):
    plt.plot(i2[i][0][0],i2[i][0][1],'g.')
for i in range(300):
    plt.plot(i3[i][0][0],i3[i][0][1],'b.')
from scipy.stats import norm
def argm(a,b,c):
    if(a>b and a>c):
        return 1
    elif (b>a and b>c):
        return 2
    elif(c>a and c>b):
        return 3
pred1=[]#计算后验概率
for i in range(600):
    pred1.append(argm(norm.pdf(i1[i][0][0],loc=1,scale=2)*norm.pdf(i1[i][0][1],loc=1,scale=2)
                      ,norm.pdf(i1[i][0][0],loc=8,scale=2)*norm.pdf(i1[i][0][1],loc=1,scale=2)
                      ,norm.pdf(i1[i][0][0],loc=4,scale=2)*norm.pdf(i1[i][0][1],loc=4,scale=2)))
pred3=[]
for i in range(300):
    pred3.append(argm(norm.pdf(i3[i][0][0],loc=1,scale=2)*norm.pdf(i3[i][0][1],loc=1,scale=2)
                      ,norm.pdf(i3[i][0][0],loc=8,scale=2)*norm.pdf(i3[i][0][1],loc=1,scale=2)
                      ,norm.pdf(i3[i][0][0],loc=4,scale=2)*norm.pdf(i3[i][0][1],loc=4,scale=2)))
pred2=[]
for i in range(100):
    pred2.append(argm(norm.pdf(i2[i][0][0],loc=1,scale=2)*norm.pdf(i2[i][0][1],loc=1,scale=2)
                      ,norm.pdf(i2[i][0][0],loc=8,scale=2)*norm.pdf(i2[i][0][1],loc=1,scale=2)
                      ,norm.pdf(i2[i][0][0],loc=4,scale=2)*norm.pdf(i2[i][0][1],loc=4,scale=2)))
print(pred1.count(1));print(pred1.count(2));print(pred3.count(3));
i1_1=np.random.multivariate_normal([1,1],I,(600,1))
i2_1=np.random.multivariate_normal([8,1],I,(300,1))
i3_1=np.random.multivariate_normal([4,4],I,(100,1))
i1_x=[]
for i in range(len(i1)):
    i1_x.append(i1[i][0][0])
i1_y=[]
for i in range(len(i1)):
    i1_y.append(i1[i][0][1])
i2_x=[]
for i in range(len(i2)):
    i2_x.append(i2[i][0][0])
i2_y=[]
for i in range(len(i2)):
    i2_y.append(i2[i][0][1])
i3_x=[]
for i in range(len(i3)):
    i3_x.append(i3[i][0][0])
i3_y=[]
for i in range(len(i3)):
    i3_y.append(i3[i][0][1])
def get_kde(x,data_array,h=2):#高斯核函数密度估计
    def gauss(x):
        import math
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2))
    N=len(data_array)
    res=0
    if len(data_array)==0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x-data_array[i])/h)
    res /= (N*h)
    return res
pred1=[]
for i in range(600):
    temp_x=i1_x[i]
    temp_y=i1_y[i]
    pred1.append(argm(get_kde(temp_x,i1_x)*get_kde(temp_y,i1_y)
                      ,get_kde(temp_x,i2_x)*get_kde(temp_y,i2_y)
                      ,get_kde(temp_x,i3_x)*get_kde(temp_y,i3_y)))

pred2=[]
for i in range(100):
    temp_x=i2_x[i]
    temp_y=i2_y[i]
    pred1.append(argm(get_kde(temp_x,i1_x)*get_kde(temp_y,i1_y)
                      ,get_kde(temp_x,i2_x)*get_kde(temp_y,i2_y)
                      ,get_kde(temp_x,i3_x)*get_kde(temp_y,i3_y)))
pred3=[]
for i in range(300):
    temp_x=i3_x[i]
    temp_y=i3_y[i]
    pred1.append(argm(get_kde(temp_x,i1_x)*get_kde(temp_y,i1_y)
                      ,get_kde(temp_x,i2_x)*get_kde(temp_y,i2_y)
                      ,get_kde(temp_x,i3_x)*get_kde(temp_y,i3_y)))
print(pred1.count(1));print(pred2.count(2));print(pred3.count(3));