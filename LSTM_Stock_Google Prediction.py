import keras 
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
Train=pd.read_csv('Google_Stock_Price_Train.csv')
Train_open=Train.iloc[:,[1]].values
Train_open=SC.fit_transform(Train_open)
x_train=[]
y_train=[]

for i in range(60,1258):
    x_train.append(Train_open[i-60:i,0])
    y_train.append(Train_open[i,0])
x_train=np.array(x_train)
y_train=np.array(y_train)

#adding extra dimensions
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=Sequential()
model.add(LSTM(units=120,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=10,return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=100,batch_size=2)
Test=pd.read_csv('Google_Stock_Price_Test.csv')
total=pd.concat((Train['Open'],Test['Open']),axis=0)
inputs=total[len(total)-len(Test)-60:].values

inputs=inputs.reshape(-1,1)
inputs=SC.transform(inputs)

x_test=[]

for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test=np.array(x_test)

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted=SC.inverse_transform(model.predict(x_test))
Test=np.array(Test)
Test=Test.iloc[:,[1]].values

import matplotlib.pyplot as plt
plt.plot(predicted,color='red',label='predicted_stock',)
plt.plot(Test,label='actual')
plt.legend()
plt.xlabel('Time')
plt.xlabel('Stock')