import random
import numpy as np
from utils import evaluate_hyperparams


# Class for Genetic Algorithm optimizer
class GA:
    def __init__(self, hyperparameter_space, pop_size, num_generations, num_parents, crossover_rate, mutation_rate):
        self.hyperparameter_space = hyperparameter_space
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    # Initialize the population with random individuals
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            # Randomly choose hyperparameters for each individual
            individual = {key: random.choice(values) for key, values in self.hyperparameter_space.items()}
            population.append(individual)
        return population

    # Select parents for crossover based on their fitness
    def selection(self, population, fitnesses):
        # Ensure all fitness scores are positive by shifting them
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        parents = random.choices(population, weights=fitnesses, k=self.num_parents)
        return parents

    # Perform crossover to generate offspring
    def crossover(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and random.random() < self.crossover_rate:
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

    # Mutate an individual's hyperparameters with a given probability
    def mutation(self, individual):
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                individual[key] = random.choice(self.hyperparameter_space[key])
        return individual

    # Replace the old population with the new offspring
    def replacement(self, population, offspring):
        population[:] = offspring
        return population

    # Main function to run the Genetic Algorithm
    def __call__(self, data, *args, **kwargs):
        population = self.initialize_population()
        classification = kwargs.get('classification', False)  # Default to False if not provided
        best_individual = None
        best_fitness = float('-inf')
        best_model = None
        his_best_f = np.zeros(self.num_generations)

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1} start")
            # Evaluate the fitness of each individual in the population
            fitnesses_models = [evaluate_hyperparams(ind, data, classification=classification, CV=True) for ind in population]
            fitnesses = [(-1) * fm[0] for fm in
                         fitnesses_models]  # Extract fitness values, negate the loss so the model with the lowest loss is the most fit
            models = [fm[1] for fm in fitnesses_models]  # Extract models
            best_gen_fitness = max(fitnesses)
            best_gen_individual = population[fitnesses.index(best_gen_fitness)]
            best_gen_model = models[fitnesses.index(best_gen_fitness)]

            if best_gen_fitness > best_fitness: # Update the best individual and fitness if the current generation is better
                best_fitness = best_gen_fitness
                best_individual = best_gen_individual
                best_model = best_gen_model
                print(f"Best individual fitness = {-best_gen_fitness}")

            parents = self.selection(population, fitnesses) # Select parents for the next generation
            offspring = self.crossover(parents) # Generate offspring through crossover and mutation
            offspring = [self.mutation(child) for child in offspring]
            population = self.replacement(population, offspring)
            his_best_f[generation] = -best_fitness # Record the best fitness in the current generation
            print(f'Generation {generation + 1}: Best Fitness = {-best_fitness}')

        return best_individual, best_model, -best_fitness, his_best_f # Return the best individual, the best model, the best fitness, and the history of best fitness values

