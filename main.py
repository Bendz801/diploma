# main.py
import json
import pandas as pd
from data_loader import load_data
from feature_engineering import create_features
from monte_carlo import run_monte_carlo
from vae_model import build_vae, train_vae
from lstm_model import train_lstm, predict_lstm
from evaluation import evaluate_predictions

def main():
    # Загружаем конфигурацию
    with open("config.json", "r") as f:
        config = json.load(f)

    # Шаг 1: Загрузка данных
    print("Загрузка данных...")
    data = load_data(config)

    # Шаг 2: Обработка данных
    print("Обработка данных...")
    features = create_features(data, config)

    # Шаг 3: Monte Carlo симуляции
    print("Запуск Monte Carlo симуляций...")
    monte_carlo_paths = run_monte_carlo(features, config)

    # Шаг 4: Обучение VAE
    print("Обучение VAE...")
    vae = build_vae(config)
    train_vae(vae, features, config)

    # Шаг 5: Обучение LSTM и прогнозирование
    print("Обучение LSTM...")
    lstm_model, scaler = train_lstm(features, config)
    predictions = predict_lstm(lstm_model, features, scaler, config)
    pd.DataFrame(predictions, columns=['Predicted']).to_csv(config["results"]["predictions_path"])

    # Шаг 6: Оценка результатов
    print("Оценка результатов...")
    evaluate_predictions(features, predictions, config)

if __name__ == "__main__":
    main()