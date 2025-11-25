import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def create_sequences(data, seq_len=60, target_col=0):
    # data: numpy array (n_obs, n_features)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_col])
    X = np.stack(X).astype(float)
    y = np.stack(y).astype(float)
    return X, y

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

def mae(a, b):
    return np.mean(np.abs(a - b))

def mape(a, b, eps=1e-8):
    return np.mean(np.abs((a - b) / (b + eps))) * 100.0
