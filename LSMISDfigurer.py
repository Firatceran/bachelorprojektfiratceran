import numpy as np
import matplotlib.pyplot as plt

# Parametrene (Den kan altså godt være anderledes men vi har brugt dem igennem hele projektet, så vi vil fortsætte med at antage samme værdier :=)
S0 = 36
r = 0.06
sigma = 0.2
T = 1
alpha = 5
n_paths = 30
n_steps = 100
n_ISD = 100000

# ISD kerne funktion
def K_ISD(U):
    return 2 * np.sin(np.arcsin(2 * U - 1) / 3)

# ISD startværdier
U = np.random.rand(n_paths)
X0 = S0 + alpha * K_ISD(U)

# Vi skal altså nu lave tidsinterval og "trin" :)
dt = T / n_steps
t_grid = np.linspace(0, T, n_steps + 1)

# Simuler prisstier med GBM
paths = np.zeros((n_paths, n_steps + 1))
for i in range(n_paths):
    W = np.random.normal(0, np.sqrt(dt), n_steps).cumsum()
    W = np.insert(W, 0, 0)
    paths[i] = X0[i] * np.exp((r - 0.5 * sigma ** 2) * t_grid + sigma * W)

# Figur 1 i afsnittet om LSM ISD (AFSNIT 7)
plt.figure(figsize=(10, 5))
for i in range(n_paths):
    plt.plot(t_grid, paths[i], linewidth=1, alpha=0.8)
plt.title("ISD", fontsize=13)
plt.xlabel("Tid")
plt.ylabel("Stock prisene")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Figur 2 i afsnittet om LSM ISD (AFSNIT 7)
U_large = np.random.rand(n_ISD)
X0_large = S0 + alpha * K_ISD(U_large)


plt.figure(figsize=(10, 5))
plt.hist(X0_large, bins=40, color='royalblue', edgecolor='black')
plt.title("ISD-Fordeling Diagram", fontsize=13)
plt.xlabel("Initial Stock Priserne")
plt.ylabel("Frekvens")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()