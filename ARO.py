import numpy as np

from utils import space_bound, evaluate_hyperparams


class ARO:
    def __init__(self, bounds, max_it, n_pop):
        self.max_it = max_it
        self.n_pop = n_pop
        self.low, self.up = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        self.dim = len(bounds)

    def __call__(self, data, *args, **kwargs):
        print("Initiating ARO Optimizer")
        best_f = float('inf')
        best_x = None
        best_model = None
        his_best_f = np.zeros(self.max_it)

        pop_pos = np.random.rand(self.n_pop, self.dim) * (self.up - self.low) + self.low
        pop_fit = np.zeros(self.n_pop)

        classification = kwargs.get('classification', False)  # Default to False if not provided

        for i in range(self.n_pop):
            print("Evaluating initial population")
            pop_fit[i], model = evaluate_hyperparams(pop_pos[i, :], data, classification=classification, CV=True)
            if pop_fit[i] <= best_f:
                best_f = pop_fit[i]
                best_x = pop_pos[i, :]
                best_model = model

        for it in range(self.max_it):
            direct1 = np.zeros((self.n_pop, self.dim))
            direct2 = np.zeros((self.n_pop, self.dim))
            theta = 2 * (1 - it / self.max_it)

            for i in range(self.n_pop):
                curr_x = pop_pos[i, :]

                L = (np.exp(1) - np.exp(((it - 1) / self.max_it) ** 2)) * np.sin(2 * np.pi * np.random.rand())
                rd = np.ceil(np.random.rand() * self.dim).astype(int)
                direct1[i, np.random.permutation(self.dim)[:rd]] = 1
                c = direct1[i, :]
                R = L * c

                A = 2 * np.log(1 / np.random.rand()) * theta

                if A > 1:
                    K = np.setdiff1d(np.arange(self.n_pop), i)
                    rand_ind = np.random.choice(K)
                    new_pop_pos = pop_pos[rand_ind, :] + R * (curr_x - pop_pos[rand_ind, :]) + np.round(
                        0.5 * (0.05 + np.random.rand())) * np.random.randn(self.dim)
                else:
                    direct2[i, np.floor(np.random.rand() * self.dim).astype(int)] = 1
                    gr = direct2[i, :]
                    H = ((self.max_it - it + 1) / self.max_it) * np.random.randn()
                    b = curr_x + H * gr * curr_x
                    new_pop_pos = curr_x + R * (np.random.rand() * b - curr_x)

                new_pop_pos = space_bound(new_pop_pos, self.up, self.low)
                new_pop_pos = new_pop_pos.ravel()
                new_pop_fit, new_model = evaluate_hyperparams(new_pop_pos, data, classification=classification, CV=True)

                if new_pop_fit < pop_fit[i]:
                    pop_fit[i] = new_pop_fit
                    pop_pos[i, :] = new_pop_pos
                    if new_pop_fit < best_f:
                        best_f = new_pop_fit
                        best_x = new_pop_pos
                        best_model = new_model
            his_best_f[it] = best_f

        return best_x, best_model, best_f, his_best_f
