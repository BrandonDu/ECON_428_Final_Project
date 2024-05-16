import numpy as np
import matplotlib.pyplot as plt
from LSTM import evaluate_lstm_models

def fun_range(fun_index):
    dim = 30

    if fun_index == 1:
        low, up = -100, 100
    elif fun_index == 2:
        low, up = -10, 10
    elif fun_index == 3:
        low, up = -100, 100
    elif fun_index == 4:
        low, up = -100, 100
    elif fun_index == 5:
        low, up = -30, 30
    elif fun_index == 6:
        low, up = -100, 100
    elif fun_index == 7:
        low, up = -1.28, 1.28
    elif fun_index == 8:
        low, up = -500, 500
    elif fun_index == 9:
        low, up = -5.12, 5.12
    elif fun_index == 10:
        low, up = -32, 32
    elif fun_index == 11:
        low, up = -600, 600
    elif fun_index == 12:
        low, up = -50, 50
    elif fun_index == 13:
        low, up = -50, 50
    elif fun_index == 14:
        low, up = -65.536, 65.536
        dim = 2
    elif fun_index == 15:
        low, up = -5, 5
        dim = 4
    elif fun_index == 16:
        low, up = -5, 5
        dim = 2
    elif fun_index == 17:
        low, up = [-5, 0], [10, 15]
        dim = 2
    elif fun_index == 18:
        low, up = -2, 2
        dim = 2
    elif fun_index == 19:
        low, up = 0, 1
        dim = 3
    elif fun_index == 20:
        low, up = 0, 1
        dim = 6
    elif fun_index == 21:
        low, up = 0, 10
        dim = 4
    elif fun_index == 22:
        low, up = 0, 10
        dim = 4
    elif fun_index == 23: # bounds and dim for lstm hyperparameters
        bounds = [(10, 100),  # LSTM units
                  (0.1, 0.5),  # Dropout rate
                  (0.001, 0.01)]  # Learning rate (assuming optimizer uses it)
        low, up = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        dim = len(bounds)

    else:
        low, up = 0, 10
        dim = 4

    return low, up, dim


def space_bound(X, Up, Low):
    Dim = len(X)
    S = (X > Up) + (X < Low)
    X = (np.random.rand(1, Dim) * (Up - Low) + Low) * S + X * (~S)
    return X


def ben_functions(X, FunIndex, Dim):
    if FunIndex == 1:  # Sphere
        return np.sum(X**2)
    elif FunIndex == 2:  # Schwefel 2.22
        return np.sum(np.abs(X)) + np.prod(np.abs(X))
    elif FunIndex == 3:  # Schwefel 1.2
        Fit = 0
        for i in range(Dim):
            Fit += np.sum(X[0:i+1])**2
        return Fit
    elif FunIndex == 4:  # Schwefel 2.21
        return np.max(np.abs(X))
    elif FunIndex == 5:  # Rosenbrock
        return np.sum(100*(X[1:Dim] - X[0:Dim-1]**2)**2 + (X[0:Dim-1] - 1)**2)
    elif FunIndex == 6:  # Step
        return np.sum(np.floor((X + 0.5))**2)
    elif FunIndex == 7:  # Quartic
        return np.sum(np.arange(1, Dim + 1) * (X**4)) + np.random.rand()
    elif FunIndex == 8:  # Schwefel
        return np.sum(-X * np.sin(np.sqrt(np.abs(X))))
    elif FunIndex == 9:  # Rastrigin
        return np.sum(X**2 - 10 * np.cos(2 * np.pi * X)) + 10 * Dim
    elif FunIndex == 10:  # Ackley
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / Dim)) - \
               np.exp(np.sum(np.cos(2 * np.pi * X)) / Dim) + 20 + np.exp(1)
    elif FunIndex == 11:  # Griewank
        return np.sum(X**2) / 4000 - np.prod(np.cos(X / np.sqrt(np.arange(1, Dim + 1)))) + 1
    elif FunIndex == 12:  # Penalized
        a, k, m = 10, 100, 4
        return (np.pi / Dim) * (10 * ((np.sin(np.pi * (1 + (X[0] + 1) / 4)))**2) +
               np.sum((((X[0:Dim-1] + 1) / 4)**2) * (1 + 10 * (np.sin(np.pi * (1 + (X[1:Dim] + 1) / 4)))**2)) +
               (((X[Dim-1] + 1) / 4)**2) + np.sum(k * ((X - a)**m) * (X > a) + k * ((-X - a)**m) * (X < -a)))
    elif FunIndex == 13:  # Penalized2
        a, k, m = 10, 100, 4
        return 0.1 * ((np.sin(3 * np.pi * X[0]))**2 + np.sum((X[0:Dim-1] - 1)**2 * (1 + (np.sin(3 * np.pi * X[1:Dim]))**2)) +
                      ((X[Dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * X[Dim-1]))**2) +
                      np.sum(k * ((X - a)**m) * (X > a) + k * ((-X - a)**m) * (X < -a)))
    elif FunIndex == 14:  # Foxholes
        a = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                      [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16]])
        b = np.zeros(25)
        for j in range(25):
            b[j] = np.sum((X - a[:, j])**6)
        return (1 / 500 + np.sum(1 / (np.arange(1, 26) + b)))**(-1)
    elif FunIndex == 15:  # Kowalik
        a = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
        b = 1 / np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        return np.sum((a - ((X[0] * (b**2 + X[1] * b)) / (b**2 + X[2] * b + X[3])))**2)
    elif FunIndex == 16:  # Six Hump Camel
        return 4 * (X[0]**2) - 2.1 * (X[0]**4) + (X[0]**6) / 3 + X[0] * X[1] - 4 * (X[1]**2) + 4 * (X[1]**4)
    elif FunIndex == 17:  # Branin
        return (X[1] - (X[0]**2) * 5.1 / (4 * (np.pi**2)) + 5 / np.pi * X[0] - 6)**2 + \
               10 * (1 - 1 / (8 * np.pi)) * np.cos(X[0]) + 10
    elif FunIndex == 18:  # GoldStein-Price
        return (1 + (X[0] + X[1] + 1)**2 * (19 - 14 * X[0] + 3 * (X[0]**2) - 14 * X[1] +
               6 * X[0] * X[1] + 3 * X[1]**2)) * \
               (30 + (2 * X[0] - 3 * X[1])**2 * (18 - 32 * X[0] + 12 * (X[0]**2) + 48 * X[1] -
               36 * X[0] * X[1] + 27 * X[1]**2))
    elif FunIndex == 19:  # Hartman 3
        a = np.array([[3, 10, 30], [.1, 10, 35], [3, 10, 30], [.1, 10, 35]])
        c = np.array([1, 1.2, 3, 3.2])
        p = np.array([[.3689, .117, .2673], [.4699, .4387, .747], [.1091, .8732, .5547], [.03815, .5743, .8828]])
        Fit = 0
        for i in range(4):
            Fit -= c[i] * np.exp(-np.sum(a[i, :] * ((X - p[i, :])**2)))
        return Fit
    elif FunIndex == 20:  # Hartman 6
        af = np.array([[10, 3, 17, 3.5, 1.7, 8], [.05, 10, 17, .1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, .05, 10, .1, 14]])
        cf = np.array([1, 1.2, 3, 3.2])
        pf = np.array([[.1312, .1696, .5569, .0124, .8283, .5886], [.2329, .4135, .8307, .3736, .1004, .9991],
                       [.2348, .1415, .3522, .2883, .3047, .6650], [.4047, .8828, .8732, .5743, .1091, .0381]])
        Fit = 0
        for i in range(4):
            Fit -= cf[i] * np.exp(-np.sum(af[i, :] * ((X - pf[i, :])**2)))
        return Fit
    elif FunIndex == 21:  # Shekel 5
        a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9],
                      [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        Fit = 0
        for i in range(5):
            Fit -= 1 / (np.dot((X - a[i, :]), (X - a[i, :]).T) + c[i])
        return Fit
    elif FunIndex == 22:  # Shekel 7
        a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9],
                      [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        Fit = 0
        for i in range(7):
            Fit -= 1 / (np.dot((X - a[i, :]), (X - a[i, :]).T) + c[i])
        return Fit
    elif FunIndex == 23: # LSTM model performance case
        return evaluate_lstm_model(X)
    else:  # Shekel 10
        a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9],
                      [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        Fit = 0
        for i in range(10):
            Fit -= 1 / (np.dot((X - a[i, :]), (X - a[i, :]).T) + c[i])
        return Fit


def ARO(F_index, MaxIt, nPop):
    # Define function range
    Low, Up, Dim = fun_range(F_index)

    # Initialize population
    PopPos = np.random.rand(nPop, Dim) * (Up - Low) + Low # nPop number of vectors of dimension "Dim" with lower and upper bounds defined
    PopFit = np.zeros(nPop)

    # Evaluate initial population
    for i in range(nPop):
        PopFit[i] = ben_functions(PopPos[i, :], F_index, Dim)

    # Initialize best fitness and solution
    BestF = float('inf')
    BestX = None

    # Find initial best fitness and solution
    for i in range(nPop):
        if PopFit[i] <= BestF:
            BestF = PopFit[i]
            BestX = PopPos[i, :]

    # Initialize history of best fitness
    HisBestF = np.zeros(MaxIt)

    # Main loop
    for It in range(MaxIt):
        Direct1 = np.zeros((nPop, Dim))
        Direct2 = np.zeros((nPop, Dim))
        theta = 2 * (1 - It / MaxIt)

        for i in range(nPop):
            L = (np.exp(1) - np.exp(((It - 1) / MaxIt) ** 2)) * np.sin(2 * np.pi * np.random.rand())
            rd = np.ceil(np.random.rand() * Dim).astype(int)
            Direct1[i, np.random.permutation(Dim)[:rd]] = 1
            c = Direct1[i, :]
            R = L * c

            A = 2 * np.log(1 / np.random.rand()) * theta

            if A > 1:
                K = np.setdiff1d(np.arange(nPop), i)
                RandInd = np.random.choice(K)
                newPopPos = PopPos[RandInd, :] + R * (PopPos[i, :] - PopPos[RandInd, :]) \
                            + np.round(0.5 * (0.05 + np.random.rand())) * np.random.randn(Dim)
            else:
                Direct2[i, np.floor(np.random.rand() * Dim).astype(int)] = 1
                gr = Direct2[i, :]
                H = ((MaxIt - It + 1) / MaxIt) * np.random.randn()
                b = PopPos[i, :] + H * gr * PopPos[i, :]
                newPopPos = PopPos[i, :] + R * (np.random.rand() * b - PopPos[i, :])

            newPopPos = space_bound(newPopPos, Up, Low)
            newPopFit = ben_functions(newPopPos, F_index, Dim)

            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos[i, :] = newPopPos

        # Update best fitness and solution
        for i in range(nPop):
            if PopFit[i] < BestF:
                BestF = PopFit[i]
                BestX = PopPos[i, :]

        HisBestF[It] = BestF

    return BestX, BestF, HisBestF


MaxIteration = 50
PopSize = 10
FunIndex = 23

# BestX, BestF, HisBestF = ARO(FunIndex, MaxIteration, PopSize)
best_hyperparams, best_error, history = ARO(F_index=23, MaxIt=50, nPop=10)


print(f'The best fitness of F{FunIndex} is: {best_error}')

plt.figure()
if best_error > 0:
    plt.semilogy(history, 'r', linewidth=2)
else:
    plt.plot(history, 'r', linewidth=2)


plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title(f'F{FunIndex}')
plt.show()