import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from data_loader import load_sp500_data, load_vix_data, load_news_data
from feature_engineering import create_features
from monte_carlo import run_monte_carlo
from vae_model import build_vae, train_vae
from lstm_model import train_lstm, predict_lstm
from evaluation import evaluate_predictions

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    try:
        sp500 = load_sp500_data(config)
        vix = load_vix_data(config)
        news = load_news_data(config)

        # Убедимся, что индексы являются DatetimeIndex
        sp500.index = pd.to_datetime(sp500.index)
        vix.index = pd.to_datetime(vix.index)
        news.index = pd.to_datetime(news.index)

        # Создаем признаки для sp500
        sp500_with_features = create_features(sp500.copy())

        # Сохраняем обработанные датасеты (опционально)
        sp500_with_features.to_csv(config['data']['processed_data_path'].replace('.csv', '_sp500.csv'))
        vix.to_csv(config['data']['processed_data_path'].replace('.csv', '_vix.csv'))
        news.to_csv(config['data']['processed_data_path'].replace('.csv', '_news.csv'))

        # --- Дальнейший пайплайн без слияния ---

        # Пример: Monte Carlo только на данных S&P 500
        run_monte_carlo(sp500_with_features.copy(), config)

        # Пример: Обучение VAE на признаках S&P 500
        vae_data = sp500_with_features.drop(columns=['Close'], errors='ignore').values
        if vae_data.size > 0:
            vae_data = (vae_data - vae_data.min(axis=0)) / (vae_data.max(axis=0) - vae_data.min(axis=0))
            vae = build_vae(config)
            train_vae(vae, vae_data, config)
        else:
            print("Предупреждение: Нет данных для обучения VAE (S&P 500).")

        # Пример: Обучение LSTM на данных S&P 500 (только временные признаки)
        lstm_features_sp500 = [col for col in config['lstm']['features'] if col in sp500_with_features.columns and col != 'sentiment']
        if lstm_features_sp500:
            lstm_data_sp500 = sp500_with_features[lstm_features_sp500].copy()
            lstm_model, scaler = train_lstm(lstm_data_sp500, config, lstm_features_sp500)
            preds = predict_lstm(lstm_model, lstm_data_sp500, scaler, config, lstm_features_sp500)
            pd.DataFrame(preds, columns=['Predicted']).to_csv(config['results']['predictions_path'].replace('.csv', '_sp500.csv'))
            evaluate_predictions(sp500_with_features.copy(), preds, config)
        else:
            print("Предупреждение: Нет доступных признаков для обучения LSTM (S&P 500).")

        print("Пайплайн выполнен (без слияния)!")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == '__main__':
    main()