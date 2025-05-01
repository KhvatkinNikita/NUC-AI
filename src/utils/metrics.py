from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def pe(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true) * 100

def mape(y_true, y_pred):
    """Calculate MAPE while avoiding division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx]))

def rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def sMAPE(y_true, y_pred):
    """Calculate SMAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    nonzero_idx = denominator != 0
    return np.mean(np.abs(y_true[nonzero_idx] - y_pred[nonzero_idx]) / denominator[nonzero_idx])

def all_metrics(y_true, y_pred):
    """Compute all relevant regression metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'sMAPE': sMAPE(y_true, y_pred)
    }


