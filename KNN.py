#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Mushroom
# dataset link : http://archive.ics.uci.edu/ml/datasets/Mushroom
# email : amirsh.nll@gmail.com


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

d = pd.read_csv('Mushroom.csv')
d.head()

x = d.drop('type',axis=1)
y = d.type

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=100)

xi=[1,3,7,9]
accuracy=[]
for i in range(len(xi)):
    knn = KNeighborsClassifier(n_neighbors=xi[i])
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
    print("accuracy n =",xi[i]," is : ",accuracy[i])

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(xi,accuracy)
plt.xlabel("Values For K")
plt.ylabel("Accuracy")

