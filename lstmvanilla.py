# univariate data preparation
import pandas as pd
import numpy as np
from numpy import array
np.random.seed(1234)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  mean_absolute_error
from matplotlib import pyplot as plt
#get data
def getData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=0)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = getData('sugar.csv')
raw_seq= raw_seq['y']

#Making data stationary
# for x in range(0, len(raw_seq)-1):
# 	raw_seq[x]=raw_seq[x+1]-raw_seq[x]
# raw_seq[len(raw_seq)-1]=0

# choose a number of time steps
n_steps = 3

# split into samples
X, y = split_sequence(raw_seq, n_steps)


# for i in range(len(X)):
# 	print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

#Splitting into testing and training
NumberOfElements = len(X)
TrainingSize = int(NumberOfElements * 0.7)
TrainingX = X[0:TrainingSize]
TrainingX=TrainingX.reshape(TrainingX.shape[0], TrainingX.shape[1], n_features)
TestX = X[TrainingSize:NumberOfElements]
Trainingy= y[0:TrainingSize]
Testy= y[TrainingSize:NumberOfElements]

Predictions= list()

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
ActualX = TrainingX
#print(ActualX)
Actualy= Trainingy
model.fit(ActualX, Actualy, epochs=200, verbose=0)
#print(Actualy)

# fit model
for timepoint in range(0, len(TestX)):
	ActualX=X[0:TrainingSize+timepoint]
	ActualValue =  TestX[timepoint]

# 	#forcast value
	x_input = ActualValue
	x_input = x_input.reshape((1, n_steps, n_features))
	Prediction = model.predict(x_input, verbose=0)
	print('%f) Actual= %f Prediction= %f' % (timepoint, Testy[timepoint], Prediction[0][0]))

# 	#add it in the list
	Predictions.append(Prediction[0][0])
	Actualy=np.append(Actualy, Testy[timepoint])
	#print(len(Predictions), len(Testy))

Error = mean_squared_error(Testy, Predictions)

print('Test Mean Squared Error (smaller the better fit): %.10f' % Error)
print("Mean Absolute Error (smaller the better fit):  %.10f" % mean_absolute_error(Testy, Predictions))

plt.figure(num='LSTM Vanilla')
plt.plot(Testy)
#plt.plot(Predictions)

plt.show()
#print(yhat)
# summarize the data
# for i in range(len(X)):
# 	print(X[i], y[i])