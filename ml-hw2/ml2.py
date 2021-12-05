##作业一内容代码，从原ipynb文件黏贴
import pandas as pd
X=pd.read_csv("H:\ml-hw2\dataset_regression.csv" )
X_tn=X[0:7]
y=X_tn['y'].values
x=X_tn['x'].values
x_t=X['x'].values
y_t=X['y'].values
import numpy as np
def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)
a0,a1 = linear_regression(x,y)
import matplotlib.pyplot as plt
plt.plot(x_t,y_t)
plt.plot(x_l,y_l)
plt.show()
xl=np.linspace(-2,2,1000)
y_l=x_l*a1+a0
print(1.5*a1+a0-y_t[7])
print(2*a1+a0-y_t[8])
for i in range(7):
    print((-2+0.5*i)*a1+a0-y_t[i])
###作业二内容代码
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv("H:\\ml-hw2\\winequality-white.csv")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
d=data.values
X=data.drop(['quality'],axis=1)
x_tn=X_train.values
Y=data.iloc[:,[11]]
y_tn=Y_train.values
x_ts=X_test.values
y_ts=Y_test.values
def h(theta, X):
    return np.dot(X, theta)
def Loss(theta, X, Y):
    m = len(X)
    return np.sum(np.dot((h(theta,X)-Y).T , (h(theta,X)-Y)) / (2 * m))
def bgd(alpha, maxloop, epsilon, X, Y,xt,yt):
    
    m,n = X.shape 
    theta = np.zeros((n,1)) 
    
    count = 0 # 记录迭代轮次
    converged = False # 是否已经收敛的标志
    error = np.inf 
    errors = [Loss(theta, X, Y),] # 记录每一次迭代得代价函数值
    error_t=[]
    print(errors)
    while count<=maxloop:
        if(converged):
            break
        count+=1
        for j in range(n):
            deriv = sum(np.dot(X[:,j].T, (h(theta, X) - Y)))/m
            theta[j] = theta[j] - alpha*deriv
        error = Loss(theta, X, Y)
        pred=np.dot(xt,theta)
        error_t.append(np.dot((pred-yt).T,(pred-yt))[0])
        print(error)
        errors.append(error)
        if(abs(errors[-1] - errors[-2]) < epsilon):
            converged = True
    return theta,errors,error_t
    t,e,et=bgd(0.00003,100000,0.0000001,x_tn,y_tn,x_ts,y_ts)
    h(t,x[0])
    plt.plot(et)