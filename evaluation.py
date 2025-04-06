# evaluation.py
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(data, predictions, config):
    """
    Оценивает прогнозы.
    Args:
        data (pd.DataFrame): Реальные данные.
        predictions (np.ndarray): Прогнозы.
        config (dict): Конфигурация.
    """
    lookback = config["lstm"]["lookback"]
    real_prices = data['Close'].values[lookback:lookback+len(predictions)]

    # Метрики
    mse = mean_squared_error(real_prices, predictions)
    mae = mean_absolute_error(real_prices, predictions)

    # Сохранение результатов
    with open(config["results"]["metrics_path"], "w") as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    data = pd.read_csv(config["data"]["processed_data_path"], parse_dates=['Date'], index_col='Date')
    predictions = pd.read_csv(config["results"]["predictions_path"])['Predicted'].values
    evaluate_predictions(data, predictions, config)