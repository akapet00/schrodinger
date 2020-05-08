import numpy as np 
from sklearn.metrics import mean_squared_error    
                        
def rmse(true, predict):
    """root mean squared error"""
    return np.sqrt(mean_squared_error(true, predict))

def mape(true, predict):
    """mean absolute percentage errror"""
    return np.mean(np.abs(true - predict)/(1 + np.abs(true))) * 100.0