
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error
import logging
logging.getLogger().setLevel(logging.ERROR)

def getData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=False)
raw_seq = getData('https://raw.githubusercontent.com/BrownKnight174/Sugar-data/master/sugar.csv')
NumberOfElements = len(raw_seq)

df=raw_seq.reset_index()
#Making data stationary
# for x in range(0, len(ActualData)-1):
#     ActualData[x]=ActualData[x+1]-ActualData[x]
# ActualData[len(ActualData)-1]=0

# ActualData= scaler.fit_transform(ActualData)

#Use 70% of data as training, rest 30% to Test model
TrainingSize = int(NumberOfElements * 0.7)
TrainingData = raw_seq[0:TrainingSize]
TestData = raw_seq[TrainingSize:NumberOfElements]
print(len(TestData))
predictionSize= len(TestData)
m= Prophet()
m.fit(TrainingData)
future= m.make_future_dataframe(periods=predictionSize)
forecast=m.predict(future)
print(forecast['yhat'].iloc[len(TrainingData):])
Error = mean_squared_error(TestData['y'], forecast['yhat'].iloc[len(TrainingData):])

print('Test Mean Squared Error (smaller the better fit): %.10f' % Error)
print("Mean Absolute Error (smaller the better fit):  %.10f" % mean_absolute_error(TestData['y'], forecast['yhat'].iloc[len(TrainingData):]))
m.plot(forecast)
m.plot_components(forecast)