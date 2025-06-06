import numpy as np
import matplotlib.pyplot as plt


# Optimeret binomial model-America put optioner
#Der er igen brugt resultater fra afsnittet om Bencmark-Binomial model
def binomial_american_put_fast(S0, K, r, T, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # Aktie priserne på maturity
    ST = S0 * d ** np.arange(N, -1, -1) * u ** np.arange(0, N + 1)
    option = np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        option = disc * (p * option[:-1] + (1 - p) * option[1:])
        ST = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1)
        option = np.maximum(K - ST, option)

    return option[0]


# Parameters
S0 = 36
K = 40
r = 0.06
sigma = 0.2
T = 1

steps = np.arange(10, 2001, 50)
option_values = [binomial_american_put_fast(S0, K, r, T, sigma, N) for N in steps]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, option_values, marker='o', linestyle='-', color='mediumseagreen', linewidth=2, markersize=4)
plt.title('Binomial Model', fontsize=14)
plt.xlabel('Tid', fontsize=12)
plt.ylabel('Option Priserne', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
