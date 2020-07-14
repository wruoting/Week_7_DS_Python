import pandas as pd
import numpy as np
from scipy.stats import linregress

class Stats:
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
        self.Y_predict = None
        self.r = None
    
    def fit(self):
        slope, intercept, r_value, p_value, std_err = linregress(self.X, self.Y)
        self.Y_predict = np.add(np.multiply(slope, self.Y), intercept)
        self.r = r_value

    def sse(self):
        squared_errors = np.square(self.Y - self.Y_predict)
        return np.sum(squared_errors)
