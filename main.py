from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from ARO import ARO
from GA import *
import tensorflow as tf
import time
from utils import *
import pandas as pd


def evaluate_optimizer(hyperparams_optimizer, optimizer_parameters, stock_data_train, stock_data_test, classification):
    predictions = {}
    actual = {}
    sequence_length = 20
    num_backtest_days = 0
    best_losses = []
    start_time = 0
    end_time = 0
    total_time = 0
    times_per_stock = []

    print(f"Initializing hyperparameters optimization for {hyperparams_optimizer}, classification = {classification}")
    for ticker, stock in stock_data_train.items():
        print(f"Training model for {ticker}")
        if hyperparams_optimizer == "ARO":
            optimizer = ARO(optimizer_parameters["bounds"], optimizer_parameters["max_iteration"], optimizer_parameters["pop_size"])
            start_time = time.time()
            best_hyperparams, best_model, best_loss, history = train_model(stock, optimizer, classification)
            end_time = time.time()
        else:
            optimizer = GA(optimizer_parameters["hyperparameter_space"], optimizer_parameters["pop_size"], optimizer_parameters["num_generations"], optimizer_parameters["num_parents"], optimizer_parameters["crossover_rate"], optimizer_parameters["mutation_rate"])
            start_time = time.time()
            best_hyperparams, best_model, best_loss, history = train_model(stock, optimizer, classification)
            end_time = time.time()

        total_time += end_time - start_time
        times_per_stock.append(end_time - start_time)
        best_losses.append(best_loss) # append best loss for each stock in order
        print(f"history = {history}")
        plt.figure(constrained_layout=True)
        plt.semilogy(range(1, len(history) + 1), history, 'r', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.title(f"{hyperparams_optimizer} Fitness for {ticker}")
        if classification:
            plt.savefig(f"Images/{hyperparams_optimizer} {ticker} Fitness Classification.png")
        else:
            plt.savefig(f"Images/{hyperparams_optimizer} {ticker} Fitness Regression.png")
        # Make predictions for the next day given a previous number of days for the entire testing data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_test_data = scaler.fit_transform(stock_data_test[ticker]['Open'][: -1].values.reshape(-1, 1)) # for backtesting, we reserve the last day to compare

        x_backtest, y_backtest = create_sequences(scaled_test_data, sequence_length, labels=None) # Create testing sequences
        predictions[ticker] = []
        for index, sequence in enumerate(x_backtest):
            # For each sequence, predict the next day and add to the dictionary
            sequence = sequence.reshape((1, sequence_length, 1))
            prediction = best_model.predict(sequence, verbose=False)
            if not classification:
                predictions[ticker].append(prediction[0][0])
            else:
                predictions[ticker].append(prediction)
        predictions[ticker] = np.array(predictions[ticker])
        actual[ticker] = scaler.inverse_transform(y_backtest)
        if not classification: # plot predicted vs actual for stock if doing regression
            predictions_reshaped = predictions[ticker].reshape(-1, 1)
            predictions_inversed = scaler.inverse_transform(predictions_reshaped)
            predictions[ticker] = predictions_inversed.ravel() # append the inversed predictions to the dictionary
            visualize_data(actual[ticker], predictions_inversed, hyperparams_optimizer, stock_name=ticker)
        num_backtest_days = len(stock_data_test[ticker]["Open"])
    # Backtesting
    # Trading strategy is high frequency, sell immediately after the Open of the predicted day, as we only predict the Open price of the next day

    initial_capital = 10000
    portfolio_backtest = {'long': {}, 'short': {}}
    cash = initial_capital
    portfolio_value_over_time = []

    print("Initiating backtesting")
    for day in range(sequence_length, num_backtest_days - 1):
        print(f"Day {day - sequence_length}")
        curr_prices = {}
        investments = {}
        investment_total = 0
        current_predictions = {ticker: predictions[ticker][day - sequence_length] for ticker in predictions}
        current_actual = {ticker: actual[ticker][day - sequence_length] for ticker in actual}

        # Trading strategy: Buy or short based on the predictions
        avg_change = np.arange(8) * 0.025 - 0.0875
        for ticker in current_predictions:
            prediction = current_predictions[ticker]
            curr_prices[ticker] = stock_data_test[ticker]['Open'].iloc[day - 1] # We want the last day that we know
            if classification:
                expected_change = np.sum(avg_change * prediction)
            else:
                expected_change = (prediction - curr_prices[ticker]) / curr_prices[ticker]

            investments[ticker] = expected_change * initial_capital / curr_prices[ticker]
            investment_total += np.abs(investments[ticker]) # We may short

        multiplier = 0.8 * cash / investment_total
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

        # Comparison to actual on the next day (the day we predict)
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
    return portfolio_value_over_time, best_losses, total_time, times_per_stock # return portfolio value, loss information, and times


### User Settings
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

stock_list = ["AAPL", "AMZN", "DIS", "GOOGL", "JPM", "LLY", "MSFT", "NVDA", "SPOT", "XOM"]
end_date = datetime.now() - timedelta(days=730)
start_date = end_date - timedelta(days=730)

stock_data_train = fetch_latest_data(stock_list, start_date, end_date)

# Backtesting data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
stock_data_test = fetch_latest_data(stock_list, start_date, end_date)

# ARO parameters
ARO_parameters = {
    "max_iteration": 10,
    "pop_size": 4,
    "bounds": [(10, 100),  # LSTM units
              (0.1, 0.5),  # Dropout rate
              (0.001, 0.01)]  # Learning rate (assuming optimizer uses it)
}

# GA parameters
GA_parameters = {
    "hyperparameter_space": {
        "units": [10, 30, 50, 70, 100],
        "learning_rate": [0.001, 0.01, 0.1],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "pop_size": 10,
    "num_generations": 10,
    "num_parents": 3,
    "crossover_rate": 0.7,
    "mutation_rate": 0.1,
}
#
# Classification based strategy
classification = True
classification_ARO_evaluation, classification_ARO_losses, classification_ARO_total_time, classification_ARO_times_per_stock = evaluate_optimizer("ARO", ARO_parameters, stock_data_train, stock_data_test, classification)
classification_GA_evaluation, classification_GA_losses, classification_GA_total_time, classification_GA_times_per_stock = evaluate_optimizer("GA", GA_parameters, stock_data_train, stock_data_test, classification)

write_results_to_file("classification_ARO_results.txt", classification_ARO_evaluation, classification_ARO_losses, classification_ARO_total_time, classification_ARO_times_per_stock)
write_results_to_file("classification_GA_results.txt", classification_GA_evaluation, classification_GA_losses, classification_GA_total_time, classification_GA_times_per_stock)

print(classification_ARO_evaluation)
print(classification_GA_evaluation)
# Plot the results
plt.figure(constrained_layout=True)
plt.plot(classification_ARO_evaluation, label='ARO')
plt.plot(classification_GA_evaluation, label='GA')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()
plt.savefig(f"Images/ARO vs GA Classification.png")


# Table for the best losses for each stock and each optimizer
data = {
    "Ticker": stock_list,
    "LSTM-ARO": classification_ARO_losses,
    "LSTM-GA": classification_GA_losses,
}

df = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(12, 8))  # set size frame

# hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create a table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # scale the table

plt.show()

fig.savefig("table.png")

# Regression based strategy
classification = False
regression_ARO_evaluation, regression_ARO_losses, regression_ARO_total_time, regression_ARO_times_per_stock = evaluate_optimizer("ARO", ARO_parameters, stock_data_train, stock_data_test, classification)
regression_GA_evaluation, regression_GA_losses, regression_GA_total_time, regression_GA_times_per_stock = evaluate_optimizer("GA", GA_parameters, stock_data_train, stock_data_test, classification)

write_results_to_file("regression_ARO_results.txt", regression_ARO_evaluation, regression_ARO_losses, regression_ARO_total_time, regression_ARO_times_per_stock)
write_results_to_file("regression_GA_results.txt", regression_GA_evaluation, regression_GA_losses, regression_GA_total_time, regression_GA_times_per_stock)

print(regression_ARO_evaluation)
print(regression_GA_evaluation)
# Plot the results
plt.figure(constrained_layout=True)
plt.plot(regression_ARO_evaluation, label='ARO')
plt.plot(regression_GA_evaluation, label='GA')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()
plt.savefig(f"Images/ARO vs GA Regression.png")

#
# Comparison between classification and regression
plt.figure(constrained_layout=True)
plt.plot(regression_ARO_evaluation, label='ARO regression')
plt.plot(classification_ARO_evaluation, label='ARO classification')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time Using ARO optimizer')
plt.legend()
plt.show()
plt.savefig(f"Images/ARO Regression vs Classification.png")


plt.figure(constrained_layout=True)
plt.plot(regression_GA_evaluation, label='GA regression')
plt.plot(classification_GA_evaluation, label='GA classification')
plt.xlabel('Time in Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time Using GA optimizer')
plt.legend()
plt.show()
plt.savefig(f"Images/GA Regression vs Classification.png")

