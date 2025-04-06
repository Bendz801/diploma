# monte_carlo.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_monte_carlo(data, config):
    """
    Выполняет Monte Carlo симуляции.
    Args:
        data (pd.DataFrame): Данные с признаками.
        config (dict): Конфигурация.
    Returns:
        np.ndarray: Траектории цен.
    """
    # Параметры
    S0 = data['Close'].iloc[-1]  # Последняя цена
    mu = data['Returns'].mean() * 252  # Годовая доходность
    sigma = data['Returns'].std() * np.sqrt(252)  # Годовая волатильность
    T = config["monte_carlo"]["time_horizon"]
    dt = config["monte_carlo"]["time_step"]
    N = int(T / dt)
    simulations = config["monte_carlo"]["simulations"]

    # Симуляция
    np.random.seed(42)
    paths = np.zeros((N, simulations))
    paths[0] = S0

    for t in range(1, N):
        Z = np.random.normal(0, 1, simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Визуализация
    plt.plot(paths[:, :100], alpha=0.5)  # Показываем только 100 траекторий
    mean_path = np.mean(paths, axis=1)
    plt.plot(mean_path, color='black', linewidth=2, label='Средняя траектория')
    plt.title("Monte Carlo Simulation")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    return paths

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    data = pd.read_csv(config["data"]["processed_data_path"], parse_dates=['Date'], index_col='Date')
    paths = run_monte_carlo(data, config)