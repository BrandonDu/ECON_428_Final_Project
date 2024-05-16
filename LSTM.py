import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

# Define the date range
start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 1, 1)

# Fetch the data
sp500_data = pdr.get_data_yahoo('^GSPC', start=start_date, end=end_date)


# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(sp500_data['Open'].values.reshape(-1,1))

# Create sequences for training
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# LSTM Model Building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM network model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)  # Prediction of the next opening price
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training and Inference
# Fit the model
model.fit(X, y, epochs=25, batch_size=32)

# Demonstration of prediction
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting the results
import matplotlib.pyplot as plt

actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(10,6))
plt.plot(actual_prices, color='blue', label='Actual S&P 500 Opening Price')
plt.plot(predicted_prices, color='red', label='Predicted S&P 500 Opening Price')
plt.title('S&P 500 Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('S&P 500 Opening Price')
plt.legend()
plt.show()

def evaluate_lstm_model(hyperparams):
    # Example: hyperparams might include [n_units, dropout_rate, learning_rate]
    n_units, dropout_rate, learning_rate = int(hyperparams[0]), hyperparams[1], hyperparams[2]

    model = Sequential()
    model.add(LSTM(units=n_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', learning_rate=learning_rate)

    # Assuming training and validation data are predefined and globally accessible
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(x_valid)
    mse = mean_squared_error(y_valid, predictions)

    return mse  # Lower MSE is better, hence directly suitable as a fitness value

