import pandas as pd
import numpy as np

def create_features(data):
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data.dropna(inplace=True)
    return data

def merge_datasets(sp500, vix, news):
    # Убедимся, что индексы являются DatetimeIndex
    sp500.index = pd.to_datetime(sp500.index)
    vix.index = pd.to_datetime(vix.index)
    news.index = pd.to_datetime(news.index)

    # Сбрасываем MultiIndex, если он есть, в обычные столбцы и устанавливаем первый столбец как индекс
    sp500 = sp500.reset_index(drop=False).set_index(sp500.columns[0])
    vix = vix.reset_index(drop=False).set_index(vix.columns[0])
    news = news.reset_index(drop=False).set_index(news.columns[0])

    # Объединяем SP500 и VIX по дате
    merged = pd.merge(sp500, vix, left_index=True, right_index=True, how='inner', suffixes=('_sp500', '_vix'))

    # Сбрасываем индекс merged и снова устанавливаем первый столбец как индекс
    merged = merged.reset_index(drop=False).set_index(merged.columns[0])

    # Присоединяем столбец sentiment из news
    merged = pd.merge(merged, news[['sentiment']], left_index=True, right_index=True, how='left')

    # Заполняем пропуски
    merged['sentiment'].fillna(method='ffill', inplace=True)

    return merged