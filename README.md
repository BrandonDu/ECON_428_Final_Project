### Evaluating Hyperparameter Optimization Strategies for LSTM Networks in Algorithmic Trading
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

optimizer = ARO(optimizer_parameters["bounds"], optimizer_parameters["max_iteration"], optimizer_parameters["pop_size"])
```

Similarly, we can create an instance of the GA optimizer. The GA takes in the entire hyperparameter space as a dictionary, 
the population size, number of generations, number of parents, crossover rate, and mutation rate. Again, we give
an example below.

```python



```
