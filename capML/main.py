import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pygsheets


# FETCHING SPREADSHEET/CSV FROM GOOGLE SHEETS
SHEET_ID = '1Q1O_vjp0bDg3PRkkk224PkvbLb02kclj5ltx32WcPhQ'
SHEET_NAME = 'Sheet1'
url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'

# DATA PREPROCESSING
# ------------------
# Read the CSV file
df = pd.read_csv(url)
# Drop the 'car_enter' and 'car_exit' columns
df = df.drop(['car_enter', 'car_exit'], axis=1)
# Combine the 'date' and 'time' columns into a single column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['weekday'] = df['datetime'].dt.day_of_week
# Drop the 'date' and 'time' columns
df = df.drop(['date', 'time'], axis=1)
df = df.set_index('datetime')
# Converting total_cars to a numeric data type
df['total_cars'] = pd.to_numeric(df['total_cars'])
# Make all data type float
df = df.astype('float32')


# CREATE NEW DATAFRAME BY EXTRACTING ALL DATA FOR CURRENT DAY OF WEEK
dt = dt.now()
currWeekday = dt.weekday();
print(dt.weekday())
grouped = df.groupby(df.weekday)
current_dow = grouped.get_group(currWeekday)
current_dow = current_dow.drop(['weekday'], axis=1)


# SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ---------------------------------------------
#   - Training set consists of 80% of the dataset
#   - Test set consists of 20% of the dataset
train, test = train_test_split(current_dow, test_size=0.20, shuffle=False)


# NORMALIZE THE TRAINING AND TEST SETS
# ------------------------------------
# Normalization:
#   - prevents the dominance of some features
#   - allows for faster convergence
#   - improves accuracy
#   - helps with regularization -> technique used to prevent overfitting
#       - overfitting is when models are trained too well on training data
#         resulting in poor performance when applied to new/unseen data.
scaler = MinMaxScaler() # Transforms features by scaling each feature to a given range.
train = scaler.fit_transform(train)
test = scaler.transform(test)


# PREPARE DATA FOR LSTM
# ---------------------
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 1
X_train, y_train = create_dataset(train, train[:, -1], time_steps)
X_test, y_test = create_dataset(test, test[:, -1], time_steps)


# BUILD & TRAIN THE LSTM MODEL
# ----------------------------
model = Sequential()
# LSTM model with 50 neurons
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# Dropout is a regularization technique that helps prevent overfitting.
#   - randomly drops neurons in the layer during each iteration
#   - forces remaining neurons to learn more general features useful for prediciton rather than memorization
#   - prevents co-adaptation of neurons
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
# Dense(1) refers to a layer which connects every input with a single output neuron
#   - performs a linear transformation of the input data
#   - defines the dimensionality of the output space (linear function mapping input to output)
#   - outputs single scalar value which is great for predicting a continous value
model.add(Dense(1))
# Early stopping helps prevent overfitting by stopping training once loss and validation loss
# do not improve for a specified number epochs
#  - 'val_loss' is the metric to monitor
#  - 'patience' is the number of epochs to wait for improvement
#  - 'verbose' controls how much information to display
#  - 'mode' controls whether to look for an improvement in the 'min' or 'max' direction
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')


# COMPILE & FIT THE MODEL
# -----------------------
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stop])


# EVALUATE THE MODEL
# ------------------
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Train MSE: ', train_score)
print('Test MSE: ', test_score)


# USE THE MODEL TO MAKE PREDICTIONS
# ---------------------------------
predictions = model.predict(X_test)
# Invert the scaling of the predictions and actual values
y_pred_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))


# GRAPHS/DATA VISUALIZATION
# -------------------------
# Inverse transform the normalized data to get the real values
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions = scaler.inverse_transform(predictions)

# 1. REAL + PREDICTED: TEST DATA (last 20 % of the dataset)
# Plot the actual values and the predicted values
plt.plot(current_dow.index.values[-len(y_test):], y_test, label='Actual')
plt.plot(current_dow.index.values[-len(predictions):], predictions, label='Predicted')
plt.legend(loc='upper right')
plt.xlabel('Datetime')
plt.ylabel('Total Cars')
plt.legend()
plt.show()
plt.close()

# 2. REAL + PREDICTED (overlay)
# Plot the predicted and actual values
plt.plot(current_dow.index, current_dow['total_cars'], label='Actual')
plt.plot(current_dow.index[-len(y_test):], y_pred_inv, label='Predicted')
plt.xlabel('Datetime')
plt.ylabel('Total Cars')
plt.legend(loc='upper right')
plt.show()
plt.close()

# 3. REAL + PREDICTED
plt.plot(current_dow.iloc[:(int(len(current_dow) * 0.8)), -1], label='Actual')
plt.plot(current_dow.index[-len(y_test):], y_pred_inv, label='Predicted')
plt.legend(loc='upper right')
plt.xlabel('Datetime')
plt.ylabel('Total Cars')
plt.show()


# PRINTING/STORING REAL & PREDICTED VALUES
# ----------------------------------------
results = pd.DataFrame({'actual': y_test.reshape(-1), 'predicted': predictions.reshape(-1)})
# Add a datetime column to the results dataframe
dates = current_dow.index[-len(y_test):]
results = pd.DataFrame(data=results, columns=['actual', 'predicted'])
results['datetime'] = dates
results = results.set_index(results['datetime'])
results = results.drop(['datetime'], axis=1)


# Resample the dataframe at 30-minute intervals. 
# & calculate the mean of the predicted total number of cars within that time interval using all data from current day of week
# This is necessary for plotting data points in the frontend/mobile application.
results_resampled = results.resample('30T').mean()
results_resampled = results_resampled.reset_index()


# Print the original dataframe
print(results)
# Print resampled dataframe
print(results_resampled)


# Store values in .CSV file
results_resampled.to_csv('results.csv', index=True)

# Exporting ML results to Google Sheet
# authorization
gc = pygsheets.authorize(service_file='/Users/christopher_garceau/crested-century-368601-3cb00067a757.json')
sh = gc.open('Google Spread Sheet API')
# select the third sheet 
wks = sh[2]
# update the third sheet with df, starting at cell A1. 
wks.set_dataframe(results_resampled,(0,0))
