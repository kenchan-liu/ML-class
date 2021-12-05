import numpy as np
import pandas as pd
import operator
from sklearn.model_selection import train_test_split
w=pd.read_csv("h://semeion.data", delimiter = r"\s+",
                header=None )
X = pd.DataFrame(w)
w=w.drop([256,257,258,259,260,261,262,263,264,265], axis=1)
y=pd.DataFrame(X.iloc[:,[256,257,258,259,260,261,262,263,264,265]])
def dict_list(dic:dict):               
        keys = dic.keys()                
        values = dic.values()            
        lst = [(key,val) for  key,val in zip(keys, values)] 
        return lst
def similarity(tests,train,labels,k):    
    data_hang=train.shape[0]              
    dis=np.tile(tests,(data_hang,1))-train   
    q=np.sqrt((dis**2).sum(axis=1)).argsort()   
    print(q)
    my_dict = {}                                   
    for i in range(k):                              
        votelabel=labels[q[i]]                         
        my_dict[votelabel] = my_dict.get(votelabel,0)+1   
    sortclasscount=sorted(dict_list(my_dict),key=operator.itemgetter(1),reverse=True)
                                                        
                                                        
    return sortclasscount[0][0]                    
#X_train, X_test, y_train, y_test = train_test_split(w, y, test_size=0.05, random_state=5)
count=0
for i in range(1593):
    X_train=w.drop([i],axis=0)
    X_test=w[i]
    y_train=y.drop([i],axis=0)
    y_test=y[i]
    yq=y_train.values
    newy=np.zeros(1592)
    q=X_train.values
    newy=np.zeros(1592)
    for i in range(1592):
        for j in range(10):
            if yq[i][j]==1.0:
                newy[i]=j
                break
    a=X_test.values
    out=similarity(a[i],q,newy,5)
    lq=y_test.values

    for j in range(10):
        if lq[i][j]==1.0:
            n=i
            break
    if n!=out:
        count+=1;