# data_loader.py
import yfinance as yf
import pandas as pd
import json

def load_data(config):
    """
    Загружает данные с Yahoo Finance.
    Args:
        config (dict): Конфигурация из config.json.
    Returns:
        pd.DataFrame: Данные.
    """
    ticker = config["data"]["ticker"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_data_path = config["data"]["raw_data_path"]

    # Загружаем данные
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(raw_data_path)
    return data

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    data = load_data(config)
    print(data.head())