import numpy as np
import matplotlib.pyplot as plt
import time

def compute_exercise_boundary(S0, K, r, T, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    stock = np.zeros((N + 1, N + 1))
    option = np.zeros((N + 1, N + 1))
    boundary = []

    for j in range(N + 1):
        stock[N, j] = S0 * (u ** (N - j)) * (d ** j)
        option[N, j] = max(K - stock[N, j], 0)

    for i in range(N - 1, -1, -1):
        early_exercise_found = False
        for j in range(i + 1):
            stock[i, j] = S0 * (u ** (i - j)) * (d ** j)
            cont_value = disc * (p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1])
            exercise_value = K - stock[i, j]
            option[i, j] = max(exercise_value, cont_value)
            if not early_exercise_found and exercise_value > cont_value:
                boundary.append((i * dt, stock[i, j]))
                early_exercise_found = True
        if not early_exercise_found:
            boundary.append((i * dt, np.nan))

    return zip(*boundary)

# Parameterne
#Bemærk at vi bruger samme parametre igennem hele opgaven
S0 = 36
K = 40
r = 0.06
T = 1
sigma = 0.2
exercise_points = [50, 500, 10000]
colors = ['crimson', 'slateblue', 'darkgreen']
boundaries = []
times = []

for N in exercise_points:
    start = time.time()
    t, s = compute_exercise_boundary(S0, K, r, T, sigma, N)
    times.append(round(time.time() - start, 4))
    boundaries.append((np.array(t), np.array(s)))

# Plottet
plt.figure(figsize=(10, 6))
for i, (t, s) in enumerate(boundaries):
    plt.plot(t, s, label=f'{exercise_points[i]} pts ({times[i]} s)', color=colors[i], linewidth=2)

plt.title('Den aktiekurs, hvor det netop bliver optimalt at udnytte optionen (Binomial Model)', fontsize=14)
plt.xlabel('Tid', fontsize=12)
plt.ylabel('Stok Priserne', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Steps & Runtime')
plt.tight_layout()
plt.show()

