import pandas as pd
import  numpy as np
import os

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

print('Tensorflow Version:',tf.version.VERSION)

path='D:\Paavana\organizations-100.csv'
df=pd.read_csv(path)
result=df.dtypes
print(result)
x=df.iloc[:,1:48].values
print(x.shape)
print(x[0])
mapping={200:0,246:1}
y=df.replace(mapping)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=42)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

print('Train samples:',x_train.shape,y_train.shape)
print('Test samples:',x_test.shape,y_test.shape)
nt,nf,no==x_train.shape[0],x_train.shape[1],y_train.shape[1]
print(nt,nf,no)
  



