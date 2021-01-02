import numpy as np


def mse(true, predict):
    """Mean Square Error"""
    return np.mean((true-predict)**2)

def rmse(true, predict):
    """Root Mean Square Error"""
    return np.sqrt(mse(true, predict))

def mape(true, predict):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs(true - predict) / (1 + np.abs(true))) * 100.0
