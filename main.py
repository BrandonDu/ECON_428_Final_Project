from datetime import datetime, timedelta
from ARO import ARO
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import *

sp500_stocks = ['SPOT']
print(sp500_stocks)
end_date = datetime.now() - timedelta(days=730)
start_date = end_date - timedelta(days=730)

stock_data = {}
for symbol in sp500_stocks:
    try:
        # Yahoo Finance sometimes requires a replacement for symbols like "BRK.B" to "BRK-B"
        symbol_yf = symbol.replace('.', '-')

        # Fetch historical data
        stock_data[symbol] = yf.download(symbol_yf, start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

print("Data download completed.")
print(stock_data)
stock_data = stock_data['SPOT']
print(stock_data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Open'].values.reshape(-1, 1))

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], len(sp500_stocks)))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
training_data = (X_train, y_train)
test_data = (X_test, y_test)
data = (training_data, test_data)


MaxIteration = 1
PopSize = 2
FunIndex = 23
optim = ARO(FunIndex, MaxIteration, PopSize)
best_hyperparams, best_model, best_error, history = optim(data)
print(f"best_hyperparams = {best_hyperparams}")
print(f"best_error = {best_error}")
print(f"history = {history}")

plt.figure()
if best_error > 0:
    plt.semilogy(history, 'r', linewidth=2)
else:
    plt.plot(history, 'r', linewidth=2)

plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title(f'F{FunIndex}')
plt.show()

# Actions based on best model
sp500_stocks = ['SPOT']
end_date = datetime.now()
start_date = end_date - timedelta(days=100)

test_stock_data = {}
for symbol in sp500_stocks:
    try:
        symbol_yf = symbol.replace('.', '-')
        test_stock_data[symbol] = yf.download(symbol_yf, start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

print("Data download completed.")
print(test_stock_data)
test_stock_data = test_stock_data['SPOT']
print(test_stock_data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_data = scaler.fit_transform(test_stock_data['Open'].values.reshape(-1, 1))

sequence_length = 30

scaled_test_data = scaled_test_data.reshape((scaled_test_data.shape[1], scaled_test_data.shape[0], len(sp500_stocks)))
print(scaled_test_data.shape)

predictions = best_model.predict(scaled_test_data)  # predict on the normalized data
predicted_prices = scaler.inverse_transform(predictions)
predicted_price = predicted_prices[-1][0]
print(predicted_prices)

if predicted_price > test_stock_data['Open'].iloc[-1]:
    print("Long")
else:
    print("Short")

