import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import axes3d, Axes3D

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        return Sigmoid.forward(z) * (1 - Sigmoid.forward(z))    
