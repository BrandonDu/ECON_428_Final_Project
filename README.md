# Evaluating Hyperparameter Optimization Strategies for LSTM Networks in Algorithmic Trading
## Getting Started 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. For our implementation, we consider optimizing a 2-layer LSTM to predict stock prices
of two types: regression and classification. The regression model predicts the expected price of the next day given the past days, and the classification
classifies the percent change in stock from the current day, with categories being 8 partitions of [-0.1, 0.1] into 8 equally sized intervals of width 0.025. The hyperparameters that we wish to optimize are the number
of units in each layer of the LSTM, the dropout rate, and the learning rate. 

## Initializing a Model
First, we create an instance of an ARO. The ARO class takes in bounds on the hyperparameters it wishes to tune, the maximum number of iterations, and the population size.
The bounds should be given as a list of tuples, with the first tuple representing the bounds on the number of units, the second tuple representing 
the bounds on the dropout rate, and the third tuple representing the bounds on the learning rate. An example is given below.

```python
ARO_parameters = {
    "max_iteration": 10,
    "pop_size": 4,
    "bounds": [(10, 100),  # LSTM units
              (0.1, 0.5),  # Dropout rate
              (0.001, 0.01)]  # Learning rate (assuming optimizer uses it)
}

ARO_optimizer = ARO(ARO_parameters["bounds"], ARO_parameters["max_iteration"], ARO_parameters["pop_size"])
```

Similarly, we can create an instance of the GA optimizer. The GA takes in the entire hyperparameter space as a dictionary, 
the population size, number of generations, number of parents, crossover rate, and mutation rate. Again, we give
an example below.

```python
GA_parameters = {
    "hyperparameter_space": {
        "units": [10,30,50,70,100],
        "learning_rate": [0.001, 0.01, 0.1],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "pop_size": 10,
    "num_generations": 10,
    "num_parents": 3,
    "crossover_rate": 0.7,
    "mutation_rate": 0.1
}

optimizer = GA(GA_parameters["hyperparameter_space"], GA_parameters["pop_size"], GA_parameters["num_generations"], GA_parameters["num_parents"], GA_parameters["crossover_rate"], GA_parameters["mutation_rate"])
```
### Evaluating Hyperparameters
To evaluate the hyperparameters of a specific model, we provide the **evaluate_hyperparameters** function. The function takes in the hyperparameters as a dictionary or as a list, the data as a tuple of tuples split into training and testing data, the number of epochs (default 25), batch size (default 32), classification flag (default False), and whether or not to use 5-fold cross-validation (default False). The function returns the minimum loss found and the best model corresponding to that loss. 
```python
hyperparams =
{
"units": 50,
"dropout": 0.3,
"learning_rate": 0.05
}


training_data = (X_train, y_train)
test_data = (X_test, y_test)
data = (training_data, test_data)

loss, model = evaluate_hyperparams(hyperparams, data, epochs=10, batch_size=20, classification=True, CV=False)
```

### Trading Strategy
The trading strategy we implement is simple. We perform a trading strategy where we hold stocks for exactly one day, and each day we invest exactly $5000 (including shorts as positive investment). Under regression models, we predict the percent increase for each stock and invest money proportionate to the predicted percent change for each stock. For example, if stock A is predicted to increase by 10% and stock B is predicted to increase by 90%, the algorithm buys $500 worth of stock A and $4500 worth of stock B.


### Evaluating Optimizer
To evaluate the optimizer, we provide the **evaluate_optimizer** function. The function takes in the optimizer type as a string ("ARO" or "GA"), the parameters for the optimizer as a dictionary, the training and testing data, and a flag for whether it is classification. The function returns the portfolio value based on the trading strategy described, loss information, total time it took to train the model, and time it took to train the model per stock. The user should create an Images folder in the repository to save plots given by this function.

```python
train_end_date = datetime.now() - timedelta(days=730)
train_start_date = end_date - timedelta(days=730)
stock_data_train = fetch_latest_data(stock_list, train_start_date, train_end_date)

test_end_date = datetime.now()
test_test_date = end_date - timedelta(days=365)
stock_data_test = fetch_latest_data(stock_list, test_test_date, test_end_date)


classification_GA_evaluation, classification_GA_losses, classification_GA_total_time, classification_GA_times_per_stock = evaluate_optimizer("GA", GA_parameters, stock_data_train, stock_data_test, False)
```

### Utilities
Some helpful utility functions may be found in the utils.py module. Of note are the following:

**create_sequences** takes in data and creates an array of sliding window of specified length. 

**fetch_latest_data** takes in ticker name for a stock and downloads the data from Yahoo Finance between a given start and end date.

**visualize_data** takes in prediction and actual price for a stock and creates a pyplot

