import matplotlib.pyplot as plt
import scipy.stats as si
from GBMbase import *
from LsmLongstaffSchwartz import *

# Her igen samme parametre med nogen ekstra tilføjelser.
simnum = 100000
sigma = 0.2
mu = 0.06
N = 10
T = 1
S0 = 36
K = 40
N2 = 50


def exercisegrænse(simnum, sigma, mu, N2, T, S0, K):
    option_priser, exercisegrænse = lsm(simnum, sigma, mu, N2, T, S0, K)
    Exercisegrænse = exercisegrænse * K
    return option_priser, Exercisegrænse


option_priser, Exercisegrænse = exercisegrænse(simnum, sigma, mu, 50, T, S0, K)
Stockpristest = sim_stock_prices2(1000, 0.2, 0.06, 50, 1, 36)


def DeltaBlackSchoels(T, N, sigma, mu, K, Exercisegrænse, StockpricesDelta, N2):
    dt = 1 / N
    exercisesteps = N2 // N
    Hedgingerror = np.zeros(len(StockpricesDelta))
    for j in range(0, len(StockpricesDelta)):
        Stockprice = StockpricesDelta[j, :]
        optionalive = np.zeros(N - 1)
        for i in range(1, N):
            if Stockprice[i] > Exercisegrænse[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        HedgeTime = int(sum(optionalive)) + 1
        Delta = np.zeros(HedgeTime)
        for i in range(HedgeTime):
            Delta[i] = Deltaput(Stockprice[i], K, T, mu, sigma, i * dt)
        Cash = np.zeros(HedgeTime)
        RP = np.zeros(HedgeTime + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N))
        Cash[0] = RP[0] - Delta[0] * Stockprice[0]
        for i in range(1, HedgeTime):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[i]
            Cash[i] = RP[i] - Delta[i] * Stockprice[i]
        RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * Stockprice[HedgeTime]
        hedgerror = np.maximum(0, K - Stockprice[HedgeTime]) - RP[HedgeTime]
        timetomaturity = (N - HedgeTime) / N
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        Hedgingerror[j] = dischedgerror
    return Hedgingerror


Stockpristest = sim_stock_prices2(10000, 0.2, 0.06, 50, 1, 36)


def DeltaBlackSchoels2(T, N, sigma, mu, K, Exercisegrænse, StockpricesDelta, N2):
    dt = 1 / N
    exercisesteps = N2 // N
    HedgingerrorT2 = np.full(len(StockpricesDelta), np.nan)
    Hedgingerrorearly = np.full(len(StockpricesDelta), np.nan)
    for j in range(0, len(StockpricesDelta)):
        Stockprice = StockpricesDelta[j, :]
        optionalive = np.zeros(N - 1)
        for i in range(1, N):
            if Stockprice[i] > Exercisegrænse[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        HedgeTime = int(sum(optionalive)) + 1
        Delta = np.zeros(HedgeTime)
        for i in range(HedgeTime):
            Delta[i] = Deltaput(Stockprice[i], K, T, mu, sigma, i * dt)
        Cash = np.zeros(HedgeTime)
        RP = np.zeros(HedgeTime + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N))
        Cash[0] = RP[0] - Delta[0] * Stockprice[0]
        for i in range(1, HedgeTime):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[i]
            Cash[i] = RP[i] - Delta[i] * Stockprice[i]
        RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * Stockprice[HedgeTime]
        hedgerror = np.maximum(0, K - Stockprice[HedgeTime]) - RP[HedgeTime]
        timetomaturity = (N - HedgeTime) / N
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        if HedgeTime == 50:
            HedgingerrorT2[j] = dischedgerror
        else:
            Hedgingerrorearly[j] = dischedgerror
    return Hedgingerror, HedgingerrorT


# Delta fordeling
def distdeltaBS(T, N, sigma, mu, S0, K, simnum, N2, M):
    deltaerror = np.zeros(M)
    for i in range(M):
        deltaerror[i] = DeltaBlackSchoels(T, N, sigma, mu, S0, K, option_priser, Exerciseboundary)
    return deltaerror


# Delta med BS modellen
def Deltaputoptioner(S0, K, T, mu, sigma, t):
    d1 = (np.log(S0 / K) + (mu + (sigma ** 2) / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    Delta = -1 * si.norm.cdf(-d1, 0, 1)
    return Delta


def DeltaBlackScholess(S0, K, T, mu, sigma, t):
    Stockprices = S0 + np.arange(-S0, S0, 1)
    DeltaBlackScholess = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        delta = Deltaputoptioner(Stockprices[i - 1], K, T, mu, sigma, t)
        Delta[i - 1] = delta
    return DeltaBlackScholess
