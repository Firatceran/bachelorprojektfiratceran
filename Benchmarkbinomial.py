import numpy as np
from GBMbase import *
import matplotlib.pyplot as plt
from time import process_time


def binomialgitter(S0, T, N, sigma):
    dt = 1 / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    Stopckpriser = np.full((N + 1, N + 1), np.nan)
    for i in np.arange(T * N, -1, -1):
        Stopckpriser[0:i + 1, i] = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
    return Stopckpriser


# Bestemme optionsprisen.
def binom_base(K, S0, T, N, r, sigma):
    dt = 1 / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S = S0 * d ** (np.arange(T * N, -1, -1)) * u ** (np.arange(0, T * N + 1, 1))
    C = np.maximum(0, K - S)
    Exerciseboundary = np.zeros(N * T)

    for i in np.arange(T * N - 1, 0, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        dummy = K - S >= C
        delta = (C[1] - C[0]) / (S[1] - S[0])
        if np.any(dummy):
            last_true_index = np.where(dummy)[0][-1]
            Exerciseboundary[i - 1] = S[last_true_index]
        else:
            Exerciseboundary[i - 1] = np.nan
        C = np.maximum(C, K - S)
    S = S0 * d ** (np.arange(0, -1, -1)) * u ** (np.arange(0, 1, 1))
    C[: 1] = disc * (q * C[1: 2] + (1 - q) * C[0: 1])
    C = C[:-1]
    dummy = K - S >= C
    if np.any(dummy):
        last_true_index = np.where(dummy)[0][-1]
        Exerciseboundary[- 1] = S[last_true_index]
    else:
        Exerciseboundary[- 1] = np.nan
    C = np.maximum(C, K - S)
    Exerciseboundary[T * N - 1] = K
    return C, delta, Exerciseboundary


binom_base(36, 40, 1, 2500, 0.06, 0.2)

t1_start = process_time()
binom_base(K, S0, T, 2500, r, sigma)
t1_stop = process_time()


def tabel3():
    optionprisbinom = np.full(502, np.nan)
    for i in range(2, 502, 1):
        print(i)
        option_pris, delta, exerciseboundary = binom_base(40, 36, 1, i, 0.06, 0.2)
        optionprisbinom[i] = option_pris

#Er vigtigt ift. illust. af graffer.
error = np.full(2500, np.nan)
for i in range(2, 2199):
    error[i] = optionprice[i] - optionprice[i + 1]

N = 50
S0 = 36
T = 1
sigma = 0.2
K = 40
r = 0.06

price, delta, Stockpris = binom_base(K, S0, T, N, r, sigma)
D = Stockpris

Delta1 = Delta(36, 40, 1, 2500, 0.06, 0.2)
Delta2 = Delta(S0, K, 2, N, r, sigma)
Delta3 = Delta(S0, K, 1, 1000, r, sigma)
Delta4 = Delta(S0, K, 1, 50000, r, sigma)


# Delta eksperiment
def binom_base2(K, S0, T, N, r, sigma, dt):
    Totalsteps = N * T
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S = S0 * d ** (np.arange(Totalsteps, -1, -1)) * u ** (np.arange(0, Totalsteps + 1, 1))
    C = np.maximum(0, K - S)
    Stockpris = np.zeros(int(T / dt), )
    Stockpris[T * N - 1] = K

    for i in np.arange(Totalsteps - 1, 0, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        dummy = K - S >= C
        delta = (C[1] - C[0]) / (S[1] - S[0])
        if np.any(dummy):
            last_true_index = np.where(dummy)[0][-1]
            Stockpris[i - 1] = S[last_true_index]
        else:
            Stockpris[i - 1] = np.nan
        C = np.maximum(C, K - S)
    delta = (C[1] - C[0]) / (S[1] - S[0])
    S = S0 * d ** (np.arange(0, -1, -1)) * u ** (np.arange(0, 1, 1))
    C[: 1] = disc * (q * C[1: 2] + (1 - q) * C[0: 1])
    C = C[:-1]
    dummy = K - S >= C
    if np.any(dummy):
        last_true_index = np.where(dummy)[0][-1]
        Stockpris[- 1] = S[last_true_index]
    else:
        Stockpris[- 1] = np.nan
    C = np.maximum(C, K - S)
    return C, delta, Stockpris


StockprisDelta = sim_stock_prices2(1000, 0.2, 0.06, 50, 1, 36)
StockprisDelta1 = sim_stock_prices2(1000, sigma, mu, 10, T, S0)
StockprisDelta2 = sim_stock_prices2(1000, sigma, mu, 25, T, S0)


# Beregn delta givet et stockpris

def BinomDelta(sigma, mu, N, T, S0, N2, StockprisDelta, K):
    start = time.time()
    dt = 1 / N
    exercisesteps = N2 // N
    dt2 = 1 / N2
    HedgingerrorD2 = np.zeros(len(StockprisDelta))
    C, delta, Exerciseboundarybinom = binom_base2(K, S0, T, N2, mu, sigma, dt2)
    for j in range(0, len(StockprisDelta)):
        optionalive = np.zeros(N - 1)
        stock_price_path = StockprisDelta[j, :]
        for i in range(1, N):
            if stock_price_path[i] > Exerciseboundary[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        HedgeTime = int(sum(optionalive)) + 1
        Delta = np.zeros(HedgeTime)
        Optionprice = np.zeros(HedgeTime + 1)
        for i in range(HedgeTime):
            C, delta, Stockpris = binom_base2(K, stock_price_path[i], T, N - i, mu, sigma, dt)
            Delta[i] = delta
            Optionprice[i] = C[0]
        # Cashflow
        Optionprice[HedgeTime] = np.maximum(0, K - stock_price_path[HedgeTime])
        Cash = np.zeros(HedgeTime)
        RP = np.zeros(HedgeTime + 1)
        RP[0] = Optionprice[0]
        disc = np.exp(mu * (1 / N))
        Cash[0] = Optionprice[0] - Delta[0] * stock_price_path[0]
        for i in range(1, HedgeTime):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * stock_price_path[i]
            Cash[i] = RP[i] - Delta[i] * stock_price_path[i]
        RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * stock_price_path[HedgeTime]
        hedgerror = Optionprice[HedgeTime] - RP[HedgeTime]
        timetomaturity = (N - HedgeTime) / N
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerrorD2[j] = dischedgerror
        end = time.time()
        print(end - start)
    return Hedgingerror


# Fordelingen af errors
def distdeltabinom(sigma, mu, N, T, S0, N2, M):
    deltaerror = np.zeros(M)
    for i in range(M):
        deltaerror[i] = BinomDelta(sigma, mu, N, T, S0, N2)
    return deltaerror


# Hedgign error til tid T
def deltaerror(HedgeTime, Optionprice, mu, N, Delta, stock_price_path):
    Cash = np.zeros(HedgeTime)
    RP = np.zeros(HedgeTime + 1)
    RP[0] = Optionprice[0]
    disc = np.exp(mu * (1 / N))
    Cash[0] = Optionprice[0] - Delta[0] * stock_price_path[0]
    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + Delta[i - 1] * stock_price_path[i]
        Cash[i] = RP[i] - Delta[i] * stock_price_path[i]
    RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * stock_price_path[HedgeTime]
    hedgerror = RP[HedgeTime] - Optionprice[HedgeTime]
    timetomaturity = (N - HedgeTime) / N
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    return dischedgerror


# stock_price_path = np.array([36., 36.48642014, 37.86151866, 34.8456401, 34.73871265,
#                             34.12517739, 35.4861796, 33.37300352, 34.69088191, 36.04537846,
#                             33.22865797])
N = 10
S0 = 36
T = 1
sigma = 0.2
K = 40
r = 0.06
N2 = 2000
