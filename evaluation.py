import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(data, preds, config):
    lookback = config['lstm']['lookback']
    actual = data['Close'].values[lookback:lookback+len(preds)]
    mse = mean_squared_error(actual, preds)
    mae = mean_absolute_error(actual, preds)
    with open(config['results']['metrics_path'], 'w') as f:
        f.write(f"MSE: {mse}\nMAE: {mae}\n")
    print(f"MSE: {mse}, MAE: {mae}")