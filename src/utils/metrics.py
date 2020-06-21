import numpy as np

def mse(true, predict):
    """mean squared error"""
    return np.mean((true-predict)**2)

def rmse(true, predict):
    """root mean squared error"""
    return np.sqrt(mse(true, predict))

def mape(true, predict):
    """mean absolute percentage errror"""
    return np.mean(np.abs(true - predict)/(1 + np.abs(true))) * 100.0
