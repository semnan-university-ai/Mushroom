#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Mushroom
# dataset link : http://archive.ics.uci.edu/ml/datasets/Mushroom
# email : amirsh.nll@gmail.com


# In[4]:


import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

d = pd.read_csv('Mushroom.csv')
d.head()

x = d.drop('type',axis=1)
y = d.type

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=100)

entropy = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=4,min_samples_leaf=5)
entropy.fit(x_train,y_train)
y_pred = entropy.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("accuracy is : ",accuracy)

tree.plot_tree(entropy)

