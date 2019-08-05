#!/usr/bin/env python
# coding: utf-8

# # Foreign Exchange Rates by Rajat Bhardwaj

# In[1]:


import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import rcParams
rcParams['figure.figsize']=10,6


# In[2]:


dataset=pd.read_csv('C://Python37//FER.csv')


# In[3]:


dataset['Date']=pd.to_datetime(dataset['Date'],infer_datetime_format=True)
indexedDataset=dataset.set_index(['Date'])


# In[4]:


from datetime import datetime
indexedDataset.head(5)


# In[5]:


plt.xlabel("Date")
plt.ylabel("Value")
plt.plot(indexedDataset)


# In[6]:


rolmean=indexedDataset.rolling(window=12).mean()
rolstd=indexedDataset.rolling(window=12).std()
rolmean,rolstd


# In[7]:


orig=plt.plot(indexedDataset, color='blue',label='Original')
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,color='black',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling mean &Standard Deviation')
plt.show(block=False)


# In[8]:


indexedDataset_logScale=np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
indexedDataset_logScale


# In[9]:


movingAverage=indexedDataset_logScale.rolling(window=12).mean()
movingSTD=indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage, color='red')


# In[10]:


datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
datasetLogScaleMinusMovingAverage.head(5)

datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(5)


# In[11]:


def test_stationarity(timeseries):
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    orig=plt.plot(timeseries, color='blue',label='Original')
    mean=plt.plot(movingAverage,color='red',label='Rolling Mean')
    std=plt.plot(movingSTD,color='black',label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling mean &Standard Deviation')
    plt.show(block=False)
    
    from statsmodels.tsa.stattools import adfuller
    print("Result of Dickey fuller test")
    dftest=adfuller(timeseries['Value'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistics','p-value','#Lags used','No of observation'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key]=value
    print(dfoutput)


# In[12]:


test_stationarity(datasetLogScaleMinusMovingAverage)


# In[13]:


exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=365, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[14]:


datasetLogScaleMinusMovingExponentialDecayAverage=indexedDataset_logScale-exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)


# In[15]:


datasetLogDiffShifting=indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[16]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(indexedDataset_logScale)
trend= decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData= residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[18]:


from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(datasetLogDiffShifting, nlags=20)
lag_pacf=pacf(datasetLogDiffShifting, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('PartialAutocorrelation Function')
plt.tight_layout()


# In[19]:


from statsmodels.tsa.arima_model import ARIMA
model= ARIMA(indexedDataset_logScale, order=(2,1,2))
results_ARIMA=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting['Value'])**2))


# In[20]:


predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()


# In[21]:


predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()


# In[22]:


predictions_ARIMA_log=pd.Series(indexedDataset_logScale['Value'].iloc[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[30]:


predicitons_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predicitons_ARIMA)


# In[24]:


def num_rows(group):
    return len(group)

def num_columns(group):
    return len(group[0])
indexedDataset_logScale


# In[25]:


results_ARIMA.plot_predict(1,77)
results=results_ARIMA.forecast(steps=30)


# In[26]:


results = results_ARIMA.forecast(steps=30)    
converted_results = [(np.exp(x)) for x in [i for i in results]]


# In[27]:


converted_results


# In[ ]:





# # End of the program

# In[ ]:




