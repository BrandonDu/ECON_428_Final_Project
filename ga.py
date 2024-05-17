import random
import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def initialize_population(pop_size, hyperparameter_space):
    population = []
    for _ in range(pop_size):
        individual = {key: random.choice(values) for key, values in hyperparameter_space.items()}
        population.append(individual)
    return population


def build_lstm_model(hyperparameters, data, input_shape):
    x_train, y_train = data[0]
    x_test, y_test = data[1]
    model = keras.Sequential([ # Use same model structure as ARO
        Input(shape=(x_train.shape[1], 1)),
        LSTM(units=hyperparameters['units'], return_sequences=True),
        Dropout(hyperparameters['dropout']),
        LSTM(units=hyperparameters['units'], return_sequences=False),
        Dropout(hyperparameters['dropout']),
        Dense(units=1)
    ])
    # model = keras.Sequential()
    # for _ in range(hyperparameters['num_layers']):
    #     model.add(LSTM(hyperparameters['units'], input_shape=input_shape, return_sequences=True))
    #     model.add(Dropout(hyperparameters['dropout']))
    # model.add(LSTM(hyperparameters['units']))
    # model.add(Dense(1, activation='linear'))
    optimizer = hyperparameters['optimizer'](learning_rate=hyperparameters['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def fitness_function(hyperparameters, x_train, y_train, x_test, y_test):
    training_data = (x_train, y_train)
    test_data = (x_test, y_test)
    data = (training_data, test_data)
    model = build_lstm_model(hyperparameters, data, input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], verbose=0)
    loss = model.evaluate(x_test, y_test, verbose=0)
    return -loss  # Negate the loss because we want to maximize the fitness


def selection(population, fitnesses, num_parents):
    # Ensure all fitness scores are positive by shifting them
    min_fitness = min(fitnesses)
    if min_fitness < 0:
        fitnesses = [f - min_fitness + 1 for f in fitnesses]
    parents = random.choices(population, weights=fitnesses,
                             k=num_parents)  # selects num_parents individuals with probability proportional to fitness score
    return parents


def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents) and random.random() < crossover_rate:
            parent1 = parents[i]
            parent2 = parents[i + 1]
            crossover_point = random.randint(1, len(parent1) - 1)
            # Split the parent dictionaries into lists of items
            parent1_items = list(parent1.items())
            parent2_items = list(parent2.items())
            # Create new dictionaries by combining slices of parent dictionaries
            child1_items = parent1_items[:crossover_point] + parent2_items[crossover_point:]
            child2_items = parent2_items[:crossover_point] + parent1_items[crossover_point:]
            # Convert lists of items back into dictionaries
            child1 = dict(child1_items)
            child2 = dict(child2_items)
            offspring.extend([child1, child2])
        else:
            offspring.extend([parents[i], parents[i + 1] if i + 1 < len(parents) else parents[i]])
    return offspring


def mutation(individual, mutation_rate, hyperparameter_space):
    for key in individual.keys():
        if random.random() < mutation_rate:
            individual[key] = random.choice(hyperparameter_space[key])
    return individual


def replacement(population, offspring):
    population[:] = offspring
    return population


def genetic_algorithm(hyperparameter_space, X_train, y_train, X_val, y_val, pop_size, num_generations, num_parents,
                      crossover_rate, mutation_rate):
    population = initialize_population(pop_size, hyperparameter_space)

    best_individual = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        print(f"Generation {generation + 1} start")
        fitnesses = [fitness_function(ind, X_train, y_train, X_val, y_val) for ind in population]
        best_gen_fitness = max(fitnesses) # Get lowest negative MSE?
        best_gen_individual = population[fitnesses.index(best_gen_fitness)]

        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_individual = best_gen_individual
            print(f"best_individual fitness (MSE) = {fitnesses[fitnesses.index(best_gen_fitness)]}")

        parents = selection(population, fitnesses, num_parents)
        offspring = crossover(parents, crossover_rate)
        offspring = [mutation(child, mutation_rate, hyperparameter_space) for child in offspring]
        population = replacement(population, offspring)

        print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

    return best_individual, -best_fitness

# hyperparameter_space = {
#     'num_layers': [1, 2, 3],
#     'units': [50, 100, 150],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'batch_size': [32, 64, 128],
#     'dropout': [0.2, 0.3, 0.4],
#     'optimizer': [Adam, RMSprop],
#     'epochs': [10, 20, 30]
# }
#
# best_hyperparameters = genetic_algorithm(hyperparameter_space, X_train, y_train, X_val, y_val, pop_size=20, num_generations=10, num_parents=10, crossover_rate=0.7, mutation_rate=0.1)
# print(f'Best Hyperparameters: {best_hyperparameters}')
