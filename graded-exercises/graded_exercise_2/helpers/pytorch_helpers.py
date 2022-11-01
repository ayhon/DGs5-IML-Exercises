import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import torch

NUM_POINTS = 1000
MIN_X = -5
MAX_X = 10

def plot_function(f, min_x=MIN_X, max_x=MAX_X, min_y=None, max_y=None):
    x = torch.linspace(start=min_x, end=max_x, steps=NUM_POINTS)
    y = f(x)
    plt.plot(x,y)
    plt.grid()

def plot_descent(f, optim_trace, min_x=MIN_X, max_x=MAX_X, min_y=None, max_y=None):
    x = torch.linspace(start=min_x, end=max_x, steps=NUM_POINTS)
    y = f(x)
    plt.plot(x,y)
    
    plt.scatter(x=[p[0] for p in optim_trace],
                y=[p[1] for p in optim_trace],
                c=cm.jet(np.linspace(0,1,len(optim_trace))))
    plt.grid()

