'''
Aryan Garg
B19153
+91-8219383122
Lab 6
'''

import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def auto_correlationfunc(train, lag):
    auto_correlation = round(scipy.stats.pearsonr(train[lag:], train[:len(train) - lag])[0], 3)
    return auto_correlation

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

#1 (a)
series = pd.read_csv("datasetA6_HP.csv")
y = series["HP"].values
plt.plot(np.arange(1, 501), y)
plt.xlabel(" Index of the day")
plt.ylabel("Power Consumed")
plt.show()

# (b)
lag_sequence = series[:len(series) - 1]
y_lag = lag_sequence["HP"].values
auto_correlation = round(scipy.stats.pearsonr(y[1:], y_lag[:])[0], 3)
print("Auto correlation coefficient between the generated one day lag time sequence and the given time sequence is ",
      auto_correlation)
print()

# (c)
plt.scatter(y[1:], y_lag[:])
plt.xlabel(" Given time sequence")
plt.ylabel("One day lagged generated sequence")
plt.show()

# (d)
week = [i for i in range(1, 8)]
corr_array = []
for days in week:
    lag_sequence = series[:len(series) - days]
    y_lag = lag_sequence["HP"].values
    auto_correlation = round(scipy.stats.pearsonr(y[days:], y_lag[:])[0], 3)
    corr_array.append(auto_correlation)
plt.plot(week, corr_array)
plt.xlabel(" Lagged Values")
plt.ylabel("Correlation Coefficients")
plt.show()

# (e)
sm.graphics.tsa.plot_acf(y[:])
plt.xlabel("Lags")
plt.ylabel("Autocorrelation Coefficient")
plt.show()

# Q2
X = series
X.drop(columns=["Date"], inplace=True)
X = pd.concat([X.shift(1), X], axis=1)
X = X.values
test = np.array(X[len(X) - 250:])
test = test.T
error = round(rmse(test[0], test[1]), 3)
print("RMSE between predicted power consumed for test data and original values for test data is ", error)
print()

# Q3
# (a)
X = series.values
train, test = X[1:len(X) - 250], X[len(X) - 250:]
model = AutoReg(train, lags=5, old_names=False)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
print()
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    print()
rmse = math.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.plot(test, predictions)
plt.xlabel(" Original Test Data")
plt.ylabel(" Predicted Test Data")
plt.show()

# (b)
P = [1, 5, 10, 15, 25]
for p in P:
    model = AutoReg(train, lags=p, old_names=False)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    rmse = math.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE for p =  ' + str(p) + ' is %.3f' % rmse)

# (c)
train = np.array(train)
train = train.T
train = train[0]
lag = 1
while auto_correlationfunc(train, lag) > (2 / math.sqrt(len(train))):
    lag += 1
print("Optimal value for lag according to auto corelation equations is ",lag)
model = AutoReg(train, lags=lag, old_names=False)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
rmse = math.sqrt(mean_squared_error(test, predictions))
print('Test RMSE for p =  ' + str(lag) + ' is %.3f' % rmse)

# (d)
print("Optimal values for lag in part b and c are ", 25, " and ", 6, "respectively")
print("Lowest RMSE values for lag in part b and c are ", 4.515, " and ", 4.538, "respectively")

