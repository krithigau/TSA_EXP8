# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 03-05-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```
Read the AirPassengers dataset
```py
data = pd.read_csv('AirPassengers.csv')
```
Focus on the '#Passengers' column
```py
passengers_data = data[['#Passengers']]
```
Display the shape and the first 10 rows of the dataset
```py
print("Shape of the dataset:", passengers_data.shape)
print("First 10 rows of the dataset:")
print(passengers_data.head(10))
```
Plot Original Dataset (#Passengers Data)
```py
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original #Passengers Data')
plt.title('Original Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```
Moving Average
Perform rolling average transformation with a window size of 5 and 10
```py
rolling_mean_5 = passengers_data['#Passengers'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['#Passengers'].rolling(window=10).mean()
```
Display the first 10 and 20 vales of rolling means with window sizes 5 and 10 respectively
```py
rolling_mean_5.head(10)
rolling_mean_10.head(20)
```
Plot Moving Average
```py
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```
Perform data transformation to better fit the model
```py
data_monthly = data.resample('MS').sum()   #Month start
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),index=data.index)

```
Exponential Smoothing
```py
# The data seems to have additive trend and multiplicative seasonality
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()
```
Make predictions for one fourth of the data
```py
model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
predictions.plot(ax=ax)
ax.legend(["data_monthly", "predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')

```
### OUTPUT:

ORIGINAL DATA:

![{D7D45FEA-A278-4E03-963A-F0D8BBA45E6B}](https://github.com/user-attachments/assets/4a0c80df-573f-495f-838c-880d41bed952)


![{30FE08DC-C3F7-46F1-95B1-96D30B6E0733}](https://github.com/user-attachments/assets/1dbea12b-b6c2-454e-b166-0522abee9063)

Moving Average

![{F3F8E4DC-21D2-4056-A4F7-CB7F3F9DD981}](https://github.com/user-attachments/assets/670c4423-8e83-4d06-a04d-e6adc7bc31b0)

Exponential Smoothing

![{18DEFF87-5553-4AD6-B2CB-62549EF4C73A}](https://github.com/user-attachments/assets/294c28f5-bc67-4dbe-bf8e-d0de8d23fbac)

Prediction

![{2DD08D09-4D75-49C1-807A-62DF4C6C56F0}](https://github.com/user-attachments/assets/117e9787-bdd7-44dd-ad78-36184499af82)

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
