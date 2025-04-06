# feature_engineering.py
import json
import pandas as pd
import numpy as np

def create_features(data, config):
    """
    Создаёт признаки из сырых данных.
    Args:
        data (pd.DataFrame): Исходные данные.
        config (dict): Конфигурация.
    Returns:
        pd.DataFrame: Данные с признаками.
    """
    # Логарифмическая доходность
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Скользящее среднее (20 дней)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Волатильность (20 дней)
    data['Volatility'] = data['Returns'].rolling(window=20).std()

    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Удаляем пропуски
    data.dropna(inplace=True)

    # Сохраняем обработанные данные
    data.to_csv(config["data"]["processed_data_path"])
    return data

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    data = pd.read_csv(config["data"]["raw_data_path"], parse_dates=['Date'], index_col='Date')
    features = create_features(data, config)
    print(features.head())