import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 40
K = 40
r = 0.06
sigma = 0.2
T = 1
n_paths = 100000
n_steps = 100

# Kernel function
def K_ISD(U):
    return 2 * np.sin(np.arcsin(2 * U - 1) / 3)

# Function for ISD payoffs
def simulate_payoffs(alpha):
    U = np.random.rand(n_paths)
    X0 = S0 + alpha * K_ISD(U)
    dt = T / n_steps
    disc = np.exp(-r * T)
    Z = np.zeros(n_paths)
    for i in range(n_paths):
        W = np.random.normal(0, np.sqrt(dt), n_steps).cumsum()
        ST = X0[i] * np.exp((r - 0.5 * sigma**2) * T + sigma * W[-1])
        Z[i] = disc * max(K - ST, 0)
    return X0, Z

# for alpha = 0.5 og 25
X1, Z1 = simulate_payoffs(alpha=0.5)
X2, Z2 = simulate_payoffs(alpha=25)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

axs[0].scatter(X1, Z1, s=0.3, alpha=0.7, color="deepskyblue")
axs[0].set_xlim(39.4, 40.6)
axs[0].set_ylim(0, 10)
axs[0].set_title(r"(a) Discounted payoff vs. $X$ for $\alpha = 0.5$", fontsize=10)
axs[0].set_xlabel("Stock price", fontsize=9)
axs[0].set_ylabel("Discounted Payoff", fontsize=9)
axs[0].tick_params(axis='both', labelsize=8)
axs[0].grid(True, linestyle="--", alpha=0.3)

axs[1].scatter(X2, Z2, s=0.3, alpha=0.7, color="deepskyblue")
axs[1].set_xlim(10, 65)
axs[1].set_ylim(0, 30)
axs[1].set_title(r"(b) Discounted payoff vs. $X$ for $\alpha = 25$", fontsize=10)
axs[1].set_xlabel("Stock price", fontsize=9)
axs[1].set_ylabel("Discounted Payoff", fontsize=9)
axs[1].tick_params(axis='both', labelsize=8)
axs[1].grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
