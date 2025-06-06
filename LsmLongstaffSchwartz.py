import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
from numpy.polynomial.laguerre import lagfit, lagval
from GBMbase import *

# Her bruger vi følgende parameter og det vil vi fortsat med at gøre resten af projektet med nogen undtagelser
simnum = 250000
sigma = 0.2
mu = 0.06
N = 6
T = 1
S0 = 36
K = 40
N2 = 50
epsilon = 0.01
N0 = 10


# Vektoren
def d(mu, N, T):
    dt = 1 / N
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-mu * t)
    return disc_vector


def laguerre_basis(x):
    basis_matrix = np.zeros((len(x), 3))
    basis_matrix[:, 0] = np.exp(-x / 2)
    basis_matrix[:, 1] = np.exp(-x / 2) * (1 - x)
    basis_matrix[:, 2] = np.exp(-x / 2) * (1 - 2 * x + x ** 2 / 2)
    return basis_matrix


# LSM algoritmen
def lsm(simnum, sigma, mu, N, T, S0, K):
    disc_vector = d(mu, N, T)
    Stock_price_paths = sim_stock_prices(simnum, sigma, mu, N, T, S0)
    Stock_price_paths = np.array(Stock_price_paths) / K
    cashflow_matrix = np.zeros((simnum, (T * N) + 1))
    exerciseboundary = np.full((T * N + 1,), np.nan)
    for j in range(N * T, 0, -1):
        cashflow_matrix[:, j] = np.maximum(0, 1 - Stock_price_paths[:, j])
    # Rekrusiv regression
    for k in range(N * T, 1, -1):
        ITM = 1 - Stock_price_paths[:, k - 1] > 0
        X = Stock_price_paths[ITM, k - 1]
        Y = np.matmul(cashflow_matrix[ITM, k:(N * T) + 1],
                      np.transpose(disc_vector[:(N * T) - (k - 1)]))  # in the money cash flows

        # Polyn. basis funktion
        Basis = laguerre_basis(X)
        model = LinearRegression().fit(Basis, Y)
        Continuation = np.zeros((simnum,))
        Continuation[ITM] = model.predict(Basis)

        #Sammenlignin
        optimhold = ITM & (Continuation <= cashflow_matrix[:, k - 1])
        cashflow_matrix[optimhold, k:(N * T) + 1] = 0
        cashflow_matrix[~optimhold, k - 1] = 0

    # Bestem udnyttelsesgrænsen ved at finde alle observationer med et positivt cashflow.
    Exercise = np.where(cashflow_matrix > 0, Stock_price_paths, np.nan)
    for i in range(N * T, 0, -1):
        exerciseboundary[i] = np.nanmax(Exercise[:, i])

    # diskonteririn
    cashflow_disc = np.matmul(cashflow_matrix[:, 1:(N * T) + 1], np.transpose(disc_vector[0:N * T]))

    # Option priser
    option_priser = K * np.sum(cashflow_disc) / simnum
    return option_priser, exerciseboundary


option_priser, exerciseboundary = lsm(100000, 0.2, 0.06, 500, 1, 36, 40)


#Bestemmer prisen givet forskellige strike-priser
def tabel3(simnum):
    Strikeprise = np.array((36, 40, 44))
    PriceLSM = np.zeros((50, 3))
    for j in range(3):
        print(j)
        for i in range(50):
            option_priser, exerciseboundary = lsm(simnum, 0.2, 0.06, 50, 1, 40, Strikeprise[j])
            PriceLSM[i, j] = option_priser
    column_averagesPrice = np.mean(PriceLSM, axis=0)
    column_std_devPrice = np.std(PriceLSM, axis=0, ddof=1)
    return column_averagesPrice, column_std_devPrice


# Bruges til at bestemme gennemsnitsprisen givet forskellige tidsintervaller
def tabel2(simnum):
    timestep = np.array((10, 50, 100, 500))
    LSM1 = np.zeros((20, 4))
    for j in range(4):
        print(j)
        for i in range(20):
            option_priser, exerciseboundary = lsm(simnum, 0.2, 0.06, timestep[j], 1, 36, 40)
            LSM1[i, j] = option_priser
    column_averages = np.mean(LSM1, axis=0)
    column_std_dev = np.std(LSM1, axis=0, ddof=1)
    standard_errors = column_std_dev / np.sqrt(LSM1.shape[0])
    return standard_errors, column_averages


# Bestemmer optionsprisen for forskellige værdier af exercise pointer.
def tabel3():
    optionprice50000 = np.full(502, np.nan)
    for i in range(2, 502, 2):
        print(i)
        option_priser, exerciseboundary = lsm(50000, 0.2, 0.06, i, 1, 36, 40)
        optionprice50000[i] = option_priser


optionpricebinom1 = optionpricebinom[~np.isnan(optionpricebinom)]
optionprice500001 = optionprice50000[~np.isnan(optionprice50000)]
optionprices12 = optionprice[~np.isnan(optionprice)]
optionprice1000001 = optionprice100000[~np.isnan(optionprice100000)]


# for table 1
def table1lsm():
    for i in range(30):
        option_priser, exerciseboundary = lsm(simnum, 0.2, mu, N, 1, S0, K)
        LSMprice17[i + 20] = option_priser
        option_prise, exerciseboundary = lsm(simnum, 0.2, mu, N, 2, S0, K)
        LSMprice18[i + 20] = option_prise
        option_pris, exerciseboundary = lsm(simnum, 0.4, mu, N, 1, S0, K)
        LSMprice19[i + 20] = option_pris
        option_pri, exerciseboundary = lsm(simnum, 0.4, mu, N, 2, S0, K)
        LSMprice20[i + 20] = option_pri
        print(i)
    return LSMprice17, LSMprice18, LSMprice19, LSMprice20

#Delta delen

def DeltaLSM(S0, K, T, N, r, sigma):
    Stockprices = S0 + np.arange(-S0, S0, 1)
    Delta = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        esp = Stockprices[i - 1] * 0.01
        option_priser1, exerciseboundary = lsm(50000, 0.2, 0.06, 50, 1, Stockprices[i - 1], 40)
        option_priser2, exerciseboundary = lsm(50000, 0.2, 0.06, 50, 1, Stockprices[i - 1] + eps, 40)
        Delta[i - 1] = (option_priser2 - option_priser1) / esp
    return Delta



# Beregner delta ved tidspunkt nul givet en aktiekurs
def Pris2(Stockprices, Exerciseboundary, epsilon, simnum, sigma, mu, T, K, N):
    disc_vector = d(mu, N, T)
    Price = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        print(i)
        Eps = Stockprices[i] * epsilon
        Stock_price_paths = sim_stock_prices(100000, sigma, mu, N, T, Stockprices[i] + Eps)
        cashflow_matrix = np.zeros((simnum, (T * N) + 1))
        for j in range(1, N * T + 1, 1):
            exercise = Stock_price_paths[:, j] <= Exerciseboundary[j]
            Stock_price_paths[exercise, j + 1:] = 0
            Stock_price_paths[~exercise, j] = 0
            # for j in range(M * T, -1, -1):
        for j in range(N * T, 0, -1):
            Cashflow = Stock_price_paths[:, j] > 0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(N * T) + 1], np.transpose(disc_vector[0:N * T]))
        Price[i] = np.sum(cashflow_disc) / simnum
    return Price


def DeltaLSM(Stockprices, Exerciseboundary, epsilon, simnum, sigma, mu, T, K):
    Stockprices = S0 + np.arange(-20, S0, 1)
    LSMeps = Pris2(Stockprices, Exerciseboundary, epsilon, simnum, sigma, mu, T, K)
    LSM = Pris2(Stockprices, Exerciseboundary, 0, simnum, sigma, mu, T, K)
    Eps = Stockprices * epsilon
    Delta = (LSMeps - LSM) / Eps
    return Delta


#N_2 exercise-punkter til at estimere grænsen
#N rebalanceringspunkter
def sekantlsm(simnum, sigma, mu, N, T, S0, K, N0, epsilon, N2):
    option_priser, exerciseboundary = lsm(100000, sigma, mu, N2, T, S0, K)
    Exerciseboundary = exerciseboundary * K
    Stockprice = StockpricesDelta[9]
    LSM = np.zeros(N)
    LSMepsilon = np.zeros(N)

    disc_vector = d(mu, N0, T)

    for i in range(0, N):
        TimeToMaturity = N - i  # time to maturity
        Stock_price_paths = sim_stock_pricesflex(simnum, sigma, mu, TimeToMaturity, Stockprice[0, i], T, N0)
        cashflow_matrix = np.zeros((simnum, (T * TimeToMaturity) + 1))
        exercisesteps = 50 // N0
        for j in range(1, TimeToMaturity * T + 1, 1):
            exercise = Stock_price_paths[:, j] <= Exerciseboundary[
                exercisesteps * (N0 - TimeToMaturity) + j * exercisesteps]
            Stock_price_paths[exercise, j + 1:] = 0
            Stock_price_paths[~exercise, j] = 0
        # for j in range(M * T, -1, -1):
        for j in range(TimeToMaturity * T, 0, -1):
            Cashflow = Stock_price_paths[:, j] > 0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        # cashflow_disc = np.matmul(cashflow_matrix[:, 0:(M * T) + 1], np.transpose(disc_vector[0:M*T+1]))
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(TimeToMaturity * T) + 1],
                                  np.transpose(disc_vector[0:TimeToMaturity * T]))
        LSM[i] = np.sum(cashflow_disc) / simnum

    for i in range(0, N):
        M = N - i
        Stock_price_paths = sim_stock_pricesflex(simnum, sigma, mu, M, Stockprice[0, i] + epsilon, T, N0)
        cashflow_matrix = np.zeros((simnum, (T * M) + 1))
        exercisesteps = 50 // N0
        for j in range(1, M * T + 1, 1):
            exercise = Stock_price_paths[:, j] <= Exerciseboundary[exercisesteps * (N0 - M) + j * exercisesteps]
            Stock_price_paths[exercise, j + 1:] = 0
            Stock_price_paths[~exercise, j] = 0
        # for j in range(M * T, -1, -1):
        for j in range(M * T, 0, -1):
            Cashflow = Stock_price_paths[:, j] > 0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        # cashflow_disc = np.matmul(cashflow_matrix[:, 0:(M * T) + 1], np.transpose(disc_vector[0:M*T+1]))
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(M * T) + 1], np.transpose(disc_vector[0:M * T]))
        LSMepsilon[i] = np.sum(cashflow_disc) / simnum
    Delta = (LSMepsilon - LSM) / epsilon
    return Delta, LSM, Stockprice


def LSMdeltaresidual():
    Delta, LSM, Stockprice = sekantlsm(simnum, sigma, mu, N, T, S0, K, N0, epsilon, N2)
    Cash = np.zeros(N)
    RP = np.zeros(N + 1)
    RP[0] = LSM[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = LSM[0] - Delta[0] * Stockprice[0, 0]
    for i in range(1, N0):
        RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - Delta[i] * Stockprice[0, i]
    RP[N] = Cash[N - 1] * disc + Delta[N - 1] * Stockprice[0, N]
    LSM = np.append(LSM, np.maximum(0, K - Stockprice[0, N]))
    return LSM, RP


# LSM HEDGING ERROR FUNKTION
# Beregner optionsprisen for hvert rebalanceringspunkt ved at simulere paths (simnum) aktiepriser
def Price(Stockprice, Exerciseboundary, HedgeTime, N0, epsilon, simnum, sigma, mu, T, K, N2):
    disc_vector = d(mu, N0, T)
    Price = np.zeros(HedgeTime)
    exercisesteps = N2 // N0
    for i in range(0, HedgeTime):
        print(i)
        M = N0 - i
        Eps = Stockprice[i] * epsilon
        Stock_price_paths = sim_stock_pricesflex(simnum, sigma, mu, M, Stockprice[i] + Eps, T, N0)
        cashflow_matrix = np.zeros((simnum, (T * M) + 1))
        for j in range(1, M * T + 1, 1):
            exercise = Stock_price_paths[:, j] <= Exerciseboundary[exercisesteps * i + j * exercisesteps]
            Stock_price_paths[exercise, j + 1:] = 0
            Stock_price_paths[~exercise, j] = 0
        for j in range(M * T, 0, -1):
            Cashflow = Stock_price_paths[:, j] > 0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(M * T) + 1], np.transpose(disc_vector[0:M * T]))
        Price[i] = np.sum(cashflow_disc) / simnum
    return Price


# Eneklt hedging error funktiom der bruges til at finde hedging tid ?? Måkse.. Er ikke sikker :)
def hedgetime(Stockprice, Exerciseboundary, N2, N0):
    exercisesteps = N2 // N0
    optionalive = np.zeros(N0 - 1)
    for i in range(1, N0):
        if Stockprice[i] > Exerciseboundary[i * exercisesteps]:
            optionalive[i - 1] = 1
        else:
            break
    HedgeTime = int(sum(optionalive)) + 1
    return HedgeTime


def hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0):
    exercisesteps = N2 // N0
    Hedgingtime = np.full(len(StockpricesDelta), np.nan)
    for j in range(len(StockpricesDelta)):
        stockprice = StockpricesDelta[j, :]
        optionalive = np.zeros(N0 - 1)
        for i in range(1, N0):
            if stockprice[i] > Exerciseboundary[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        Hedgingtime[j] = int(sum(optionalive) + 1)
    return Hedgingtime


# Ibid.
def LSMsekant2(Exerciseboundary, StockpricesDelta, N0, epsilon, simnum, sigma, mu, T, K, N2):
    Hedgingtime = hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0)
    Hedgingeror = np.full(len(StockpricesDelta), np.nan)

    for j in range(800, 1000):
        print('stock path', j)
        Hedgingtimej = int(Hedgingtime[j])
        print('Hedgingtime', Hedgingtimej)
        StockpricesDeltaj = StockpricesDelta[j]
        LSM = Price(StockpricesDeltaj, Exerciseboundary, Hedgingtimej, N0, 0, 100000, sigma, mu, T, K, N2)
        LSMepsilon = Price(StockpricesDeltaj, Exerciseboundary, Hedgingtimej, N0, epsilon, 50000, sigma, mu, T, K, N2)
        Eps = StockpricesDeltaj * epsilon
        Delta = (LSMepsilon - LSM) / Eps[:Hedgingtimej]
        Delta = np.clip(Delta, -1, 0)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - Delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * StockpricesDeltaj[i]
            Cash[i] = RP[i] - Delta[i] * StockpricesDeltaj[i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + Delta[Hedgingtimej - 1] * StockpricesDeltaj[Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        Hedgingeror[j] = dischedgerror
    return Hedgingeror


def LSMsekant2(Exerciseboundary, StockpricesDelta, N0, epsilon, simnum, sigma, mu, T, K, N2):
    Hedgingtime = hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0)
    HedgingerorLSM100000 = np.full(len(StockpricesDelta), np.nan)

    for j in range(100, 200):
        print('stock path', j)
        Hedgingtimej = int(Hedgingtime[j])
        print('Hedgingtime', Hedgingtimej)
        StockpricesDeltaj = StockpricesDelta[j]
        LSM = Price(StockpricesDeltaj, Exerciseboundary, Hedgingtimej, N0, 0, 100000, sigma, mu, T, K, N2)
        LSMepsilon = Price(StockpricesDeltaj, Exerciseboundary, Hedgingtimej, N0, epsilon, 100000, sigma, mu, T, K, N2)
        Eps = StockpricesDeltaj * epsilon
        Delta = (LSMepsilon - LSM) / Eps[:Hedgingtimej]
        Delta = np.clip(Delta, -1, 0)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - Delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * StockpricesDeltaj[i]
            Cash[i] = RP[i] - Delta[i] * StockpricesDeltaj[i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + Delta[Hedgingtimej - 1] * StockpricesDeltaj[Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerorLSM100000[j] = dischedgerror
    return HedgingerorLSM100000

#Hedge tid igen
def hedgetime(Stockprice, Exerciseboundary, N2, N0, sigma, mu, T, S0, K):
    exercisesteps = N2 // N0
    optionalive = np.zeros(N0 - 1)
    for i in range(1, N0):
        if Stockprice[i] > Exerciseboundary[i * exercisesteps]:
            optionalive[i - 1] = 1
        else:
            break
    HedgeTime = int(sum(optionalive)) + 1  # deltahedge tiden
    return HedgeTime

#Hedging error funktion
def LSMsekant(HedgeTime, Exerciseboundary, Stockprice, N0, epsilon, simnum, sigma, mu, T, K):
    LSM = Price(Stockprice, Exerciseboundary, HedgeTime, N0, 0, simnum, sigma, mu, T, K, N2)
    LSMepsilon = Price(Stockprice, Exerciseboundary, HedgeTime, N0, epsilon, simnum, sigma, mu, T, K, N2)
    Eps = np.zeros(HedgeTime)
    for i in range(0, HedgeTime):
        Eps[i] = Stockprice[0, i] * epsilon
    Delta = (LSMepsilon - LSM) / Eps
    Cash = np.zeros(HedgeTime)
    RP = np.zeros(HedgeTime + 1)
    RP[0] = LSM[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = LSM[0] - Delta[0] * Stockprice[0, 0]
    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - Delta[i] * Stockprice[0, i]
    RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * Stockprice[0, HedgeTime]
    LSM = np.append(LSM, np.maximum(0, K - Stockprice[0, HedgeTime]))
    hedgerror = RP[HedgeTime] - LSM[HedgeTime]
    timetomaturity = (N0 - HedgeTime) / N0
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    return dischedgerror
 #Ikke anvendt
    deltaerror = np.zeros(100)
    for i in range(100):
        HedgeTime, Exerciseboundary, Stockprice = hedgetime(exerciseboundary, N2, N0, sigma, mu, T, S0, K)
        deltaerror[i] = LSMsekant(HedgeTime, Exerciseboundary, Stockprice, N0, epsilon, simnum, sigma, mu, T, K)
        print(i)
    return deltaerror


def LSMdeltaresidual():
    Delta, LSM, Stockprice = sekantlsm(simnum, sigma, mu, N, T, S0, K, N0, epsilon)
    Cash = np.zeros(N)
    RP = np.zeros(N + 1)
    RP[0] = LSM[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = LSM[0] - Delta[0] * Stockprice[0, 0]
    for i in range(1, N0):
        RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - Delta[i] * Stockprice[0, i]
    RP[N] = Cash[N - 1] * disc + Delta[N - 1] * Stockprice[0, N]
    LSM = np.append(LSM, np.maximum(0, K - Stockprice[0, N]))
    return LSM, RP


LSMprice1
LSM1column_averages = np.mean(LSMprice1[0:50], axis=0)
LSM1column_std_dev = np.std(LSMprice1[0:50], axis=0, ddof=1)

LSMprice2
LSM2column_averages = np.mean(LSMprice2[0:50], axis=0)
LSM2column_std_dev = np.std(LSMprice2[0:50], axis=0, ddof=1)

LSMprice3
LSM3column_averages = np.mean(LSMprice3[0:50], axis=0)
LSM3column_std_dev = np.std(LSMprice3[0:50], axis=0, ddof=1)

LSMprice4
LSM3column_averages = np.mean(LSMprice4[0:50], axis=0)
LSM3column_std_dev = np.std(LSMprice4[0:50], axis=0, ddof=1)

LSMprice5, LSMprice6, LSMprice7, LSMprice8

LSMprice9
LSM9column_averages = np.mean(LSMprice9[0:50], axis=0)
LSM9column_std_dev = np.std(LSMprice9[0:50], axis=0, ddof=1)

LSMprice10
LSM10column_averages = np.mean(LSMprice10[0:50], axis=0)
LSM10column_std_dev = np.std(LSMprice10[0:50], axis=0, ddof=1)

LSMprice11
LSM11column_averages = np.mean(LSMprice11[0:50], axis=0)
LSM11column_std_dev = np.std(LSMprice11[0:50], axis=0, ddof=1)

LSMprice12
LSM12column_averages = np.mean(LSMprice12[0:50], axis=0)
LSM12column_std_dev = np.std(LSMprice12[0:50], axis=0, ddof=1)

LSMprice13, LSMprice14, LSMprice15, LSMprice16

LSMprice17
LSM17column_averages = np.mean(LSMprice17[0:50], axis=0)
LSM17column_std_dev = np.std(LSMprice17[0:50], axis=0, ddof=1)

LSMprice18
LSM18column_averages = np.mean(LSMprice18[0:50], axis=0)
LSM18column_std_dev = np.std(LSMprice18[0:50], axis=0, ddof=1)

LSMprice19
LSM19column_averages = np.mean(LSMprice19[0:50], axis=0)
LSM19column_std_dev = np.std(LSMprice19[0:50], axis=0, ddof=1)

LSMprice20
LSM19column_averages = np.mean(LSMprice20[0:50], axis=0)
LSM19column_std_dev = np.std(LSMprice20[0:50], axis=0, ddof=1)

t1_start = process_time()
option_priser, exerciseboundary = lsm(100000, 0.2, 0.06, 50, 1, 36, 40)
t1_stop = process_time()
t1_stop - t1_start
