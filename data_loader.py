import yfinance as yf
import pandas as pd
import json

def load_sp500_data(config):
    ticker = config["data"]["ticker_sp500"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_data_path = config["data"]["raw_data_path"]

    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(raw_data_path)
    return data

def load_vix_data(config):
    ticker = config["data"]["ticker_vix"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    vix_data_path = config["data"]["vix_data_path"]

    vix_data = yf.download(ticker, start=start_date, end=end_date)
    vix_data.to_csv(vix_data_path)
    return vix_data

def load_news_data(config):
    path = config["data"]["news_data_path"]
    date_col = 'pub_date'

    usecols = [date_col]
    df_head = pd.read_csv(path, nrows=0)
    if 'sentiment' in df_head.columns:
        usecols.append('sentiment')
    elif 'news_desk' in df_head.columns:
        usecols.append('news_desk')

    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=100_000,
    ):
        chunks.append(chunk)

    df = pd.concat(chunks)

    # Преобразуем столбец даты в Datetime, некорректные значения станут NaT
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')

    # Удаляем строки с NaT в столбце даты
    df.dropna(subset=[date_col], inplace=True)

    # Устанавливаем 'pub_date' в качестве индекса
    df.set_index(date_col, inplace=True)

    if 'sentiment' not in df.columns and 'news_desk' in df.columns:
        df['sentiment'] = df['news_desk'].astype('category').cat.codes.astype(float) / df['news_desk'].nunique() - 0.5
    elif 'sentiment' not in df.columns:
        df['sentiment'] = 0.0

    return df