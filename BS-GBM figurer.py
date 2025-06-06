import numpy as np
import matplotlib.pyplot as plt

# Funktion-antitetiske
# De anvendte ligninger er taget direkte fra afsnittet om Black-Scholes :)
def generate_paths_antithetic(num_paths, volatility, drift, steps_per_year, years, initial_price):
    dt = 1 / steps_per_year
    paths = np.zeros((num_paths, years * steps_per_year + 1))
    paths[:, 0] = initial_price

    for i in range(0, num_paths, 2):
        for t in range(1, years * steps_per_year + 1):
            z = np.random.normal()
            increment = (drift - 0.5 * volatility**2) * dt + volatility * z * np.sqrt(dt)
            paths[i, t] = paths[i, t-1] * np.exp(increment)
            paths[i+1, t] = paths[i+1, t-1] * np.exp((drift - 0.5 * volatility**2) * dt - volatility * z * np.sqrt(dt))
    return paths

# Funktion-uden antitetiske stier
def generate_paths_standard(num_paths, volatility, drift, steps_per_year, years, initial_price):
    dt = 1 / steps_per_year
    paths = np.zeros((num_paths, years * steps_per_year + 1))
    paths[:, 0] = initial_price

    for i in range(num_paths):
        for t in range(1, years * steps_per_year + 1):
            z = np.random.normal()
            paths[i, t] = paths[i, t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * z * np.sqrt(dt))
    return paths

#Uden antithetic
def plot_gbm_basic(num_paths, sigma, mu, steps, T, S0):
    data = generate_paths_standard(num_paths, sigma, mu, steps, T, S0)
    time_points = np.linspace(0, T, steps + 1)

    plt.figure(figsize=(10, 6))
    for path in data:
        plt.plot(time_points, path, lw=1)
    plt.title('Simulerede aktiekurser (uden antithetic)', fontsize=14)
    plt.xlabel('Tid (år)')
    plt.ylabel('Aktiekurs')
    plt.grid(True)
    plt.show()

#Med antithetic
def plot_gbm_antithetic(num_paths, sigma, mu, steps, T, S0):
    data = generate_paths_antithetic(num_paths, sigma, mu, steps, T, S0)
    time_points = np.linspace(0, T, steps + 1)

    plt.figure(figsize=(10, 6))
    for path in data:
        plt.plot(time_points, path, linestyle='--', alpha=0.7)
    plt.title('Simulerede aktiekurser (med antithetic)', fontsize=14)
    plt.xlabel('Tid (år)')
    plt.ylabel('Aktiekurs')
    plt.grid(True)
    plt.show()

plot_gbm_basic(5, 0.2, 0.06, 200, 1, 36)
plot_gbm_antithetic(6, 0.2, 0.06, 50, 1, 36)
