#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Mushroom
# dataset link : http://archive.ics.uci.edu/ml/datasets/Mushroom
# email : amirsh.nll@gmail.com


# In[5]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

d = pd.read_csv('Mushroom.csv')
d.head()

x = d.drop('type',axis=1)
y = d.type

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=100)

m = GaussianNB()
m.fit(x_train,y_train)

y_pred = m.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("accuracy is : ",accuracy)

