from datetime import datetime, timedelta
from ARO import ARO
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from GA import *
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from utils import *


def evaluate_optimizer(hyperparams_optimizer, optimizer_parameters, stock_data_train, stock_dat_test, classification):
    predictions = {}
    actual = {}
    sequence_length = 60
    num_backtest_days = 0

    print(f"Initializing hyperparameters optimization for {hyperparams_optimizer}, classification = {classification}")
    for ticker in stock_data_train:
        stock = stock_data_train[ticker]
        if hyperparams_optimizer == "ARO":
            optimizer = ARO(optimizer_parameters["fun_index"], optimizer_parameters["max_iteration"], optimizer_parameters["pop_size"])
            best_hyperparams, best_model, best_error, history = train_model(stock, optimizer, classification)
        else:
            optimizer = GA(optimizer_parameters["hyperparameter_space"], optimizer_parameters["pop_size"], optimizer_parameters["num_generations"], optimizer_parameters["num_parents"], optimizer_parameters["crossover_rate"], optimizer_parameters["mutation_rate"])
            best_hyperparams, best_model, best_fitness = train_model(stock, optimizer, classification)


        # print(f"best_hyperparams found = {best_hyperparams}")
        # print(f"best_error = {best_error}")
        # print(f"history = {history}")
        #
        # plt.figure()
        # if best_error > 0:
        #     plt.semilogy(history, 'r', linewidth=2)
        # else:
        #     plt.plot(history, 'r', linewidth=2)
        #
        # plt.xlabel('Iterations')
        # plt.ylabel('Fitness')
        # plt.title(f'F{FunIndex}')
        # plt.show()

        # Make predictions for the next day given a previous number of days for the entire testing data
        scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_test_data = scaler.fit_transform(stock_data_test['Open'].values.reshape(-1, 1))
        scaled_test_data = scaler.fit_transform(stock_data_test[ticker]['Open'][: -1].values.reshape(-1, 1)) # for backtesting, we reserve the last day to compare

        x_backtest, y_backtest = create_sequences(scaled_test_data, sequence_length, labels=None) # Create testing sequences
        scaled_test_data = scaled_test_data.reshape(
            (scaled_test_data.shape[1], scaled_test_data.shape[0], 1)
        )
        predictions[ticker] = []
        for index, sequence in enumerate(x_backtest):
            # For each sequence, predict the next day and add to the dictionary
            sequence = sequence.reshape((1, sequence_length, 1))
            prediction = best_model.predict(sequence)
            # predictions[ticker].append(prediction)
            predictions[ticker].append(prediction[0][0] if not classification else prediction)


        # predictions[ticker].append(best_model.predict(scaled_test_data))  # predict on the normalized data
        predictions[ticker] = np.array(predictions[ticker])
        print(predictions[ticker])
        actual[ticker] = scaler.inverse_transform(y_backtest)
        if not classification: # plot predicted vs actual for stock if doing regression
            predictions_reshaped = predictions[ticker].reshape(-1, 1)
            predictions_inversed = scaler.inverse_transform(predictions_reshaped)
            visualize_data(actual[ticker], predictions_inversed, stock_name=ticker)
        num_backtest_days =len(stock_data_test[ticker]["Open"])

    # Backtesting
    # Trading strategy is high frequency, sell immediately after the Open of the predicted day, as we only predict the Open price of the next day
    initial_capital = 10000
    portfolio_backtest = {'long': {}, 'short': {}}
    cash = initial_capital
    investments = {}
    investment_total = 0
    curr_prices = {}
    portfolio_value_over_time = []

    print("Initiating backtesting")
    # Loop through each trading day in the backtest period
    for day in range(sequence_length, num_backtest_days - 1):
        print(f"Day {day - sequence_length}")
        curr_prices = {}
        investments = {}
        investment_total = 0
        current_predictions = {ticker: predictions[ticker][day - sequence_length] for ticker in predictions}
        current_actual = {ticker: actual[ticker][day - sequence_length] for ticker in actual}

        # Trading strategy: Buy or short based on the predictions
        if classification:
            avg_change = np.arange(8) * 0.025 - 0.0875
            for ticker in current_predictions:
                prediction = current_predictions[ticker]
                expected_change = np.sum(avg_change * prediction)
                curr_prices[ticker] = stock_data_test[ticker]['Open'].iloc[day - 1] # We want the last day that we know
                investments[ticker] = expected_change * initial_capital / curr_prices[ticker]
                investment_total += np.abs(investments[ticker]) # We may short

            multiplier = 0.5 * cash / investment_total
            for ticker, amount in investments.items():
                amount = math.floor(amount * multiplier)
                if amount > 0: # Long
                    shares = amount // curr_prices[ticker]
                    cost = shares * curr_prices[ticker]
                    portfolio_backtest['long'][ticker] = shares
                    cash -= cost
                else: # Short
                    shares = abs(amount) // curr_prices[ticker]
                    revenue = shares * curr_prices[ticker]
                    portfolio_backtest['short'][ticker] = shares
                    cash += revenue

            # Comparison to actual
            # Calculate value of long positions
            for ticker, shares in portfolio_backtest['long'].items():
                cash += shares * current_actual[ticker][0]

            # Calculate value of short positions
            for ticker, shares in portfolio_backtest['short'].items():
                cash -= shares * current_actual[ticker][0]  # Profit or loss from short positions

            print(f"Cash after day {day - sequence_length}= {cash}")
            # Record the portfolio value for the current day
            portfolio_value_over_time.append(cash)

            # Reset the portfolio for the next day
            portfolio_backtest = {'long': {}, 'short': {}}
        else:  # Regression-based strategy
            for ticker in current_predictions:
                predicted_price = current_predictions[ticker]
                actual_price = current_actual[ticker][0]
                curr_price = stock_data_test[ticker]['Open'].iloc[day - 1]  # We want the last day that we know
                price_diff = predicted_price - curr_price

                if price_diff > 0:  # Long
                    amount = (price_diff / curr_price) * initial_capital
                    shares = amount // curr_price
                    cost = shares * curr_price
                    portfolio_backtest['long'][ticker] = shares
                    cash -= cost
                elif price_diff < 0:  # Short
                    amount = (abs(price_diff) / curr_price) * initial_capital
                    shares = amount // curr_price
                    revenue = shares * curr_price
                    portfolio_backtest['short'][ticker] = shares
                    cash += revenue

            # Comparison to actual
            # Calculate value of long positions
            for ticker, shares in portfolio_backtest['long'].items():
                cash += shares * current_actual[ticker][0]

            # Calculate value of short positions
            for ticker, shares in portfolio_backtest['short'].items():
                cash -= shares * current_actual[ticker][0]  # Profit or loss from short positions

            print(f"Cash after day {day - sequence_length}= {cash}")
            # Record the portfolio value for the current day
            portfolio_value_over_time.append(cash)

            # Reset the portfolio for the next day
            portfolio_backtest = {'long': {}, 'short': {}}


    print(f"Finished evaluating {hyperparams_optimizer} optimizer")
    # portfolio_value_over_time = [int(value[0]) for value in portfolio_value_over_time]
    return portfolio_value_over_time


### User Settings
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPOT"]
stock_list = ["SPOT"]
end_date = datetime.now() - timedelta(days=730)
start_date = end_date - timedelta(days=730)

stock_data_train = fetch_latest_data(stock_list, start_date, end_date)

# Backtesting data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
stock_data_test = fetch_latest_data(stock_list, start_date, end_date)

# ARO parameters
ARO_parameters = {
    "max_iteration": 1,
    "pop_size": 2,
    "fun_index": 23,
}

# GA parameters
GA_parameters = {
    "hyperparameter_space": {
        "units": [10, 30, 50, 70, 100],
        "learning_rate": [0.001, 0.01, 0.1],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "pop_size": 3,
    "num_generations": 3,
    "num_parents": 2,
    "crossover_rate": 0.7,
    "mutation_rate": 0.1,
}

hyperparams_optimizer = "GA" # Choose which hyperparameter optimizer to use here

# # Classification based strategy
# classification = True
# classification_ARO_evaluation = evaluate_optimizer("ARO", ARO_parameters, stock_data_train, stock_data_test, classification)
# classification_GA_evaluation = evaluate_optimizer("GA", GA_parameters, stock_data_train, stock_data_test, classification)
#
# print(classification_ARO_evaluation)
# print(classification_GA_evaluation)
# # Plot the results
# plt.figure()
# plt.plot(classification_ARO_evaluation, label='ARO')
# plt.plot(classification_GA_evaluation, label='GA')
# plt.xlabel('Time in Days')
# plt.ylabel('Portfolio Value')
# plt.title('Portfolio Value Over Time')
# plt.legend()
# plt.show()

# Regression based strategy
classification = False
regression_ARO_evaluation = evaluate_optimizer("ARO", ARO_parameters, stock_data_train, stock_data_test, classification)
regression_GA_evaluation = evaluate_optimizer("GA", GA_parameters, stock_data_train, stock_data_test, classification)

print(regression_ARO_evaluation)
print(regression_GA_evaluation)
# Plot the results
plt.figure()
plt.plot(regression_ARO_evaluation, label='ARO')
plt.plot(regression_GA_evaluation, label='GA')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()

# Comparison between classification and regression
plt.figure()
plt.plot(regression_ARO_evaluation, label='ARO regression')
plt.plot(classification_ARO_evaluation, label='ARO classification')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time Using ARO optimizer')
plt.legend()
plt.show()

plt.figure()
plt.plot(regression_GA_evaluation, label='GA regression')
plt.plot(classification_GA_evaluation, label='GA classification')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time Using GA optimizer')
plt.legend()
plt.show()

# # GA Optimizer
# hyperparameter_space = {
#     "units": [10, 30, 50, 70, 100],
#     "learning_rate": [0.001, 0.01, 0.1],
#     "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
# }
#
# pop_size = 5
# num_generations = 5
# num_parents = 2
# crossover_rate = 0.7
# mutation_rate = 0.1
#
# lp = LineProfiler()
# lp.add_function(fitness_function)
# lp.add_function(build_lstm_model)
# lp.add_function(selection)
# lp.add_function(crossover)
# lp.add_function(mutation)
# lp.add_function(replacement)
# lp_wrapper = lp(genetic_algorithm)
# best_hyperparameters, best_fitness = lp_wrapper(
#         hyperparameter_space,
#         data,
#         pop_size,
#         num_generations,
#         num_parents,
#         crossover_rate,
#         mutation_rate,
# )
# # best_hyperparameters, best_fitness = genetic_algorithm(hyperparameter_space, data, pop_size=pop_size, num_generations=num_generations, num_parents=num_parents, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
# # print(f'Best Hyperparameters: {best_hyperparameters}')
#
# lp.print_stats()
#
