import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo(data, config):
    S0 = data['Close'].iloc[-1]
    mu = data['Returns'].mean() * 252
    sigma = data['Returns'].std() * np.sqrt(252)
    T = config['monte_carlo']['time_horizon']
    dt = config['monte_carlo']['time_step']
    N = int(T / dt)
    sims = config['monte_carlo']['simulations']
    np.random.seed(42)
    paths = np.zeros((N, sims))
    paths[0] = S0
    for t in range(1, N):
        Z = np.random.normal(size=sims)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    plt.plot(paths[:, :100], alpha=0.5)
    plt.plot(paths.mean(axis=1), color='black', linewidth=2, label='Average')
    plt.title('Monte Carlo Simulation')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return paths
