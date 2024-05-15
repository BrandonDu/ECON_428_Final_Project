import numpy as np

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
    else:
        low, up = 0, 10
        dim = 4

    return low, up, dim


def space_bound(X, Up, Low):
    Dim = len(X)
    S = (X > Up) + (X < Low)
    X = (np.random.rand(1, Dim) * (Up - Low) + Low) * S + X * (~S)
    return X