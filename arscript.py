# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:09:12 2021

@author: Satyake
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt
from pylab import rcParams
data=pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=(True))
data[0:50].plot()
######
moving_Avg=data.rolling(window=5).mean()
plt.plot(moving_Avg)
###################
fit1=SimpleExpSmoothing(data).fit(smoothing_level=0.2)
fit2=SimpleExpSmoothing(data).fit(smoothing_level=0.8)
plt.plot(fit1.fittedvalues,color='blue')
plt.plot(fit2.fittedvalues,color='red')
#####################
fit1=Holt(data).fit() #Linear trend assumption
fit2=Holt(data,exponential=True).fit()
plt.plot(fit1.fittedvalues,color='blue')
plt.plot(fit2.fittedvalues,color='red')
##############################################
res=seasonal_decompose(data,model='additive')
res.plot()
plt.plot(res.seasonal)
##################################################
from statsmodels.tsa.stattools import adfuller,acf
adfuller(data) #nulhypoth time serries non stationary
#####################################################
#######Making ts stationary
z=acf(data)
plt.plot(z)
##############################
data=[10,12,14,16,18,20,20]
from statsmodels.tsa.ar_model import AutoReg,ARMA
armodel=AutoReg((data),1)
arfit=armodel.fit()
y_pred=arfit.predict(8,10)
z=data+y_pred.tolist()
plt.plot(z)
############################################
from statsmodels.tsa.arima_model import ARMA
ama=ARMA(data,order=(1,1))
ama=ama.fit()
pred=ama.predict(8,10)
z=data+pred.tolist()
plt.plot(z)












