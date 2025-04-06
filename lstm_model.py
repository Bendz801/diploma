# lstm_model.py
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(data, lookback):
    """
    Подготавливает данные для LSTM.
    Args:
        data (pd.DataFrame): Данные.
        lookback (int): Количество шагов назад.
    Returns:
        tuple: X, y для обучения.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback):
    """
    Создаёт модель LSTM.
    Args:
        lookback (int): Количество шагов назад.
    Returns:
        Sequential: Модель LSTM.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(data, config):
    """
    Обучает модель LSTM.
    Args:
        data (pd.DataFrame): Данные.
        config (dict): Конфигурация.
    Returns:
        tuple: Обученная модель, scaler.
    """
    lookback = config["lstm"]["lookback"]
    X, y, scaler = prepare_lstm_data(data, lookback)
    model = build_lstm_model(lookback)
    model.fit(X, y, epochs=config["lstm"]["epochs"], batch_size=config["lstm"]["batch_size"], verbose=1)
    return model, scaler

def predict_lstm(model, data, scaler, config):
    """
    Делает прогноз с помощью LSTM.
    Args:
        model (Sequential): Модель LSTM.
        data (pd.DataFrame): Данные.
        scaler (MinMaxScaler): Скейлер.
        config (dict): Конфигурация.
    Returns:
        np.ndarray: Прогнозы.
    """
    lookback = config["lstm"]["lookback"]
    X, _, _ = prepare_lstm_data(data, lookback)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    data = pd.read_csv(config["data"]["processed_data_path"], parse_dates=['Date'], index_col='Date')
    model, scaler = train_lstm(data, config)
    predictions = predict_lstm(model, data, scaler, config)
    pd.DataFrame(predictions, columns=['Predicted']).to_csv(config["results"]["predictions_path"])