from datetime import datetime, timedelta
from ARO import ARO
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop
from utils import *
from GA import *
from line_profiler import LineProfiler

sp500_stocks = ["SPOT"]
print(sp500_stocks)
end_date = datetime.now() - timedelta(days=730)
start_date = end_date - timedelta(days=730)

stock_data = {}
for symbol in sp500_stocks:
    try:
        # Yahoo Finance sometimes requires a replacement for symbols like "BRK.B" to "BRK-B"
        symbol_yf = symbol.replace(".", "-")

        # Fetch historical data
        stock_data[symbol] = yf.download(symbol_yf, start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

print("Data download completed.")
print(stock_data)
stock_data = stock_data["SPOT"]
print(stock_data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data["Open"].values.reshape(-1, 1))

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], len(sp500_stocks)))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
training_data = (X_train, y_train)
test_data = (X_test, y_test)
data = (training_data, test_data)

"""
# ARO Optimizer
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
"""

# GA Optimizer
hyperparameter_space = {
    # "num_layers": [1, 2, 3],
    "units": [10, 30, 50, 70, 100],
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "optimizer": [Adam, RMSprop],
    "epochs": [10, 20, 30],
}

pop_size = 5
num_generations = 5
num_parents = 2
crossover_rate = 0.7
mutation_rate = 0.1

lp = LineProfiler()
lp.add_function(fitness_function)
lp.add_function(build_lstm_model)
lp.add_function(selection)
lp.add_function(crossover)
lp.add_function(mutation)
lp.add_function(replacement)
lp_wrapper = lp(genetic_algorithm)
best_hyperparameters, best_fitness = lp_wrapper(
        hyperparameter_space,
        X_train,
        y_train,
        X_test,
        y_test,
        pop_size,
        num_generations,
        num_parents,
        crossover_rate,
        mutation_rate,
)
# best_hyperparameters, best_fitness = genetic_algorithm(hyperparameter_space, X_train, y_train, X_test, y_test, pop_size=pop_size, num_generations=num_generations, num_parents=num_parents, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
# print(f'Best Hyperparameters: {best_hyperparameters}')

lp.print_stats()


0-50
50-100
100-150
150-200
200-250

[0,0,0,0,1]
[0,1,0,0,0]

