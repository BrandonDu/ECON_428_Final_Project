import keras
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# # Function to get S&P 500 list
# def get_sp500_symbols():
#     table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#     sp500 = table[0]
#     return sp500['Symbol'].tolist()


# Create sequences for training
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def evaluate_lstm_model(hyperparams, train_data, test_data, epochs=25, batch_size=32, graphs=False):
    # Unpack hyperparameters
    n_units, dropout_rate, learning_rate = int(hyperparams[0]), hyperparams[1], hyperparams[2]
    print(f"n_units = {n_units}, dropout_rate = {dropout_rate}, learning_rate = {learning_rate}")

    # # Normalize the training data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_train_data = scaler.fit_transform(train_data['Open'].values.reshape(-1, 1))
    #
    # X_train, y_train = create_sequences(scaled_train_data, sequence_length)
    #
    # # Normalize the test data using the same scaler as training data
    # scaled_test_data = scaler.transform(test_data['Open'].values.reshape(-1, 1))
    # X_test, y_test = create_sequences(scaled_test_data, sequence_length)

    X_train, y_train = train_data[0], train_data[1]
    X_test, y_test = test_data[0], test_data[1]

    # Build the LSTM network model
    model = Sequential([
        LSTM(units=n_units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(dropout_rate),
        LSTM(units=n_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)  # Prediction of the next opening price
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(actual_prices, predicted_prices)

    if graphs:
        plt.figure(figsize=(10, 6))
        plt.plot(actual_prices, color='blue', label='Actual S&P 500 Opening Price')
        plt.plot(predicted_prices, color='red', label='Predicted S&P 500 Opening Price')
        plt.title('S&P 500 Stock Price Prediction')
        plt.xlabel('Time (days)')
        plt.ylabel('S&P 500 Opening Price')
        plt.legend()
        plt.show()

    return mse, model  # Return the mean squared error as a measure of model performance


def mean_squared_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    # Calculate the differences between actual and predicted values
    errors = actual - predicted
    # Square the errors
    squared_errors = errors ** 2
    # Calculate the mean of the squared errors
    mean_squared_error = np.mean(squared_errors)
    return mean_squared_error



# Example subset of S&P 500 stocks
# sp500_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
# sp500_stocks = get_sp500_symbols()
sp500_stocks = ['SPOT']
print(sp500_stocks)

# Calculate dates for the past year
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

stock_data = {}
for symbol in sp500_stocks:
    try:
        # Yahoo Finance sometimes requires a replacement for symbols like "BRK.B" to "BRK-B"
        symbol_yf = symbol.replace('.', '-')

        # Fetch historical data
        stock_data[symbol] = yf.download(symbol_yf, start=start_date, end=end_date)

        # Generate filename and save to CSV
        # filename = f"{output_directory}{symbol}_past_year.csv"
        # stock_data.to_csv(filename)

        # print(f"Data for {symbol} saved to {filename}")
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

print("Data download completed.")
print(stock_data)
stock_data = stock_data['SPOT']
# stock_data = pd.DataFrame(stock_data)
print(stock_data.head())

# Data Preprocessing

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Open'].values.reshape(-1, 1))

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

X = X.reshape((X.shape[0], X.shape[1], len(sp500_stocks)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
training_data = (X_train, y_train)
test_data = (X_test, y_test)
# Run LSTM Model
hyperparams = (50, 0.1, 0.001)
mse = evaluate_lstm_model(hyperparams, training_data, test_data, epochs=25, batch_size=32)
print(f"mse = {mse}")