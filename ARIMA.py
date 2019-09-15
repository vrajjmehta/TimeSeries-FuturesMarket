import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

#get data
def getData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=0)

#Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


#Get exchange rates
ActualData = getData('sugar.csv')
ActualData=  ActualData.values.astype('float32')
#Size of exchange rates
NumberOfElements = len(ActualData)
scaler= MinMaxScaler()

#Making data stationary
# for x in range(0, len(ActualData)-1):
#     ActualData[x]=ActualData[x+1]-ActualData[x]
# ActualData[len(ActualData)-1]=0

# ActualData= scaler.fit_transform(ActualData)
print(ActualData)
#Use 70% of data as training, rest 30% to Test model
TrainingSize = int(NumberOfElements * 0.7)
TrainingData = ActualData[0:TrainingSize]
TestData = ActualData[TrainingSize:NumberOfElements]


#new arrays to store actual and predictions
Actual = [x for x in TrainingData]
print(TestData)
Predictions = list()
#print(type(TestData))
#in a for loop, predict values using ARIMA model
for timepoint in range(0, len(TestData)):
    ActualValue =  TestData[timepoint]
    #forcast value
    Prediction = StartARIMAForecasting(Actual, 3,1,0)
    print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
    #add it in the list
    Predictions.append(Prediction)
    Actual.append(ActualValue)

#Print MSE to see how good the mod
TestArray= [x for x in TestData]
print(len(TestArray), len(Predictions))
Error = mean_squared_error(TestData, Predictions)

print('Test Mean Squared Error (smaller the better fit): %.10f' % Error)
print("Mean absolute error is %.10f" % mean_absolute_error(TestData, Predictions))
# plot


plt.figure(num='ARIMA')
plt.plot(TestArray)
plt.plot(Predictions)

#plt.plot(TestData.in,)
plt.show()
