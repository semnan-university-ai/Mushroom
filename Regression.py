#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Mushroom
# dataset link : http://archive.ics.uci.edu/ml/datasets/Mushroom
# email : amirsh.nll@gmail.com


# In[3]:


import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

d = pd.read_csv('Mushroom.csv')
d.head()

x = d.drop('type',axis=1)
y = d.type

x_train,x_test,y_train,y_test = train_test_split(x,y)
r = LogisticRegression(C=10.0,random_state=10,solver='liblinear')
r.fit(x_train,y_train)
y_pred = r.predict(x_test)

accuracy = metrics.accuracy_score(y_test,y_pred)
print("accuracy is : ",accuracy)

