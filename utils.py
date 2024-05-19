import math
import yfinance as yf
import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Ensures the values of X are within the given bounds (Up, Low)
def space_bound(X, Up, Low):
    Dim = len(X)
    S = (X > Up) + (X < Low)
    X = (np.random.rand(1, Dim) * (Up - Low) + Low) * S + X * (~S)  # Generate new values for out of bound elements
    return X


# Label the buckets for classification based on percent change
def label_buckets(change):
    return math.floor(max(min(change, 0.09999), -0.1) / 0.025) + 4 # Each bucket is 2.5% (i.e. 0 to 2.5% change is one bucket)

# Create sequences of data for training the LSTM model
def create_sequences(data, sequence_length, labels=None):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        if labels is None:
            y = data[i + sequence_length]  # Regression
        else:
            y = labels.iloc[i + sequence_length]  # Classification
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Calculate the mean squared error between actual and predicted values
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


# Evaluate the LSTM model hyperparameters
def evaluate_hyperparams(hyperparams, data, epochs=25, batch_size=32, classification=False, CV=False):
    # Extract hyperparameters
    if isinstance(hyperparams, dict): # Hyperparameters from GA are in dictionary format
        n_units, dropout_rate, learning_rate = hyperparams["units"], hyperparams["dropout"], hyperparams[
            "learning_rate"]
    else: # Hyperparameters from ARO are in array format
        n_units, dropout_rate, learning_rate = int(hyperparams[0]), hyperparams[1], hyperparams[2]
    num_classes = 1

    x_train, y_train = data[0]
    x_test, y_test = data[1]

    # Prepare data for classification
    if classification:
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        y_train = to_categorical(y_train, num_classes=8)
        y_test = to_categorical(y_test, num_classes=8)
        num_classes = y_train.shape[1]

    if not CV:     # No cross-validation case
        # Define the LSTM model
        model = keras.Sequential([
            Input(shape=(x_train.shape[1], 1)),
            LSTM(units=n_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=n_units, return_sequences=False),
            Dropout(dropout_rate),
        ])

        # Add output layer based on if it is classification or regression and compile model
        if classification:
            model.add(Dense(units=num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(units=1))
            loss = 'mean_squared_error'
            metrics = ['mse']

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Train and evaluate the model
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)  # Classification
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, model

    else: # CV case
        X = np.concatenate((x_train, x_test), axis=0) # Re-concatenate train and test data to be split by CV
        y = np.concatenate((y_train, y_test), axis=0)
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_loss = float("inf")
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            print(f"CV fold {fold}")
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create and train the LSTM model
            model = keras.Sequential([
                Input(shape=(x_train.shape[1], 1)),
                LSTM(units=n_units, return_sequences=True),
                Dropout(dropout_rate),
                LSTM(units=n_units, return_sequences=False),
                Dropout(dropout_rate),
            ])

            if classification:
                model.add(Dense(units=num_classes, activation='softmax'))
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(Dense(units=1))
                loss = 'mean_squared_error'
                metrics = ['mse']

            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Train and evaluate the model
            early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])  # Classification
            loss, accuracy = model.evaluate(x_test, y_test)

            # Keep track of the best model
            if loss < best_loss:
                best_loss = loss
                best_model = model

        print("Finished evaluating hyperparameters, returning best loss and model")
        return best_loss, best_model

# Visualize actual vs. predicted stock prices
def visualize_data(y_test, y_pred, optimizer, stock_name="S&P500"):
    plt.figure(constrained_layout=True, figsize=(10, 6))
    plt.plot(y_test, color='blue', label=f'Actual {stock_name} Opening Price')
    plt.plot(y_pred, color='red', label=f'Predicted {stock_name} Opening Price')
    plt.title(f'{optimizer} {stock_name} Stock Price Prediction')
    plt.xlabel('Time (days)')
    plt.ylabel(f'{stock_name} Opening Price')
    plt.legend()
    # plt.show()
    plt.savefig(f"Images/{optimizer} {stock_name}.png")


# Fetch latest stock data for given tickers and date range
def fetch_latest_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        try:
            symbol_yf = ticker.replace(".", "-")
            stock_data[ticker] = yf.download(symbol_yf, start=start_date, end=end_date)
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
    print("Stock data fetched.")
    return stock_data

# Train the LSTM model based on stock, optimizer, and classification or regression
def train_model(stock, optimizer, classification):
    scaler = MinMaxScaler(feature_range=(0, 1))

    if classification:
        percent_change = stock["Open"].pct_change().dropna()
        labels = percent_change.apply(label_buckets)
        scaled_data = scaler.fit_transform(stock["Open"].values[1:].reshape(-1, 1))
    else:
        labels = None
        scaled_data = scaler.fit_transform(stock["Open"].values.reshape(-1, 1))

    sequence_length = 20
    X, y = create_sequences(scaled_data, sequence_length, labels=labels)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    training_data = (X_train, y_train)
    test_data = (X_test, y_test)
    data = (training_data, test_data)

    return optimizer(data, classification=classification)


# Write optimizer evaluation results to a file
def write_results_to_file(file_name, evaluation, losses, total_time, times_per_stock):
    # Write the data to a text file
    with open(file_name, 'w') as file:
        file.write("classification_ARO_evaluation:\n")
        for item in evaluation:
            file.write(f"{item}\n")

        file.write("\nclassification_ARO_losses:\n")
        for item in losses:
            file.write(f"{item}\n")

        file.write("\nclassification_ARO_total_time:\n")
        file.write(f"{total_time}\n")

        file.write("\nclassification_ARO_times_per_stock:\n")
        file.write(f"Average time per stock = {np.mean(times_per_stock)}\n")
        for item in times_per_stock:
            file.write(f"{item}\n")



