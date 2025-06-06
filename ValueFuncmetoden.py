import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from LsmLongstaffSchwartz import *

simnum = 50000
sigma = 0.2
mu = 0.06
TimeToMaturity = 10
S0 = 36
T = 1
alpha = 5
N0 = 10
N = 50
K = 40


def ISD(simnum, S0, alpha):
    U = np.zeros(simnum)
    U_2 = np.random.uniform(0, 1, (simnum // 2,))
    for i in range(simnum // 2):
        U[2 * i] = U_2[i]
        U[2 * i + 1] = U_2[i]
    K_isd = 2 * np.sin(np.arcsin(2 * U - 1) / 3)
    X = S0 + alpha * K_isd
    return X


# Antihetick stier til at simulere stockpriser :)
def stockpriserISD(simnum, sigma, mu, N, S0, T, alpha):
    dt = 1 / N
    Stock_price_paths = np.zeros((simnum, (T * N) + 1))
    # Stock priser paths
    Stock_price_paths[:, 0] = ISD(simnum, S0, alpha)
    for i in range(0, simnum, 2):
        for j in range(1, (T * N) + 1):
            z = np.random.normal(loc=0, scale=1)
            Stock_price_paths[i, j] = Stock_price_paths[i, j - 1] * np.exp(
                (mu - sigma ** 2 / 2) * dt + sigma * z * np.sqrt(dt))
            Stock_price_paths[i + 1, j] = Stock_price_paths[i + 1, j - 1] * np.exp(
                (mu - sigma ** 2 / 2) * dt - sigma * z * np.sqrt(dt))
    return Stock_price_paths


def stock_priser_ISD_2(simnum, sigma, mu, N, S0, T, alpha, N0):
    dt = 1 / N0
    Stock_price_paths = np.zeros((simnum, (T * N) + 1))
    Stock_price_paths[:, 0] = ISD(simnum, S0, alpha)
    for i in range(0, simnum, 2):
        for j in range(1, (T * N) + 1):
            z = np.random.normal(loc=0, scale=1)
            Stock_price_paths[i, j] = Stock_price_paths[i, j - 1] * np.exp(
                (mu - sigma ** 2 / 2) * dt + sigma * z * np.sqrt(dt))
            Stock_price_paths[i + 1, j] = Stock_price_paths[i + 1, j - 1] * np.exp(
                (mu - sigma ** 2 / 2) * dt - sigma * z * np.sqrt(dt))
    return Stock_price_paths


# Funktion der bruges i OLS regressionen :)
def basepolynom(X, x0, degree):
    basis = [(X - x0) ** j for j in range(1, degree + 1)]
    return np.column_stack(basis)


# Den er ikke brugt i selve opgaven men er afgørende her i kode delen, da den også er anvendt i selve artiklen.
# ... maybe idk. Har ellers ikke fået den til at virke uden det.
def twostage(simnum, sigma, mu, N, T, S0, K, alpha):
    disc_vector = d(mu, N, T)
    option_priser, exerciseboundary = lsm(simnum, sigma, mu, N, T, K - 4, K)
    Stock_price_paths = stockpriserISD(simnum, sigma, mu, N, S0, T, alpha)
    Stock_price_paths = np.array(Stock_price_paths) / K
    cashflow_matrix = np.zeros((simnum, (T * N) + 1))
    for j in range(2, N * T + 1, 1):
        exercise = Stock_price_paths[:, j] <= exerciseboundary[j]
        Stock_price_paths[exercise, j + 1:] = 0
        Stock_price_paths[~exercise, j] = 0
    for j in range(N * T, 0, -1):
        Cashflow = Stock_price_paths[:, j] > 0
        cashflow_matrix[Cashflow, j] = np.maximum(0, 1 - Stock_price_paths[Cashflow, j])

    cashflow_disc = np.matmul(cashflow_matrix[:, 2:(N * T) + 1], np.transpose(disc_vector[0:(N * T) - 1]))

    # det her skal være laguerre idk ???
    X_regression = Stock_price_paths[:, 1]
    Y_regression = cashflow_disc
    Basis = laguerre_basis(X_regression)
    model = LinearRegression().fit(Basis, Y_regression)
    Continuation = model.predict(Basis)

    # Bestemmer value-funktionen på tidspunkt 1 og diskonterer den til tidspunkt 0
    Vt1 = np.maximum(Continuation, cashflow_matrix[:, 1])
    Vt1_disc = Vt1 * disc_vector[0]

    # Udfører regressionen for at bestemme (Greeks) maybee ..
    Y_OLS = K * Vt1_disc
    X = K * Stock_price_paths[:, 0]
    X_OLS = basepolynom(X, S0, 8)
    X_OLS = sm.add_constant(X_OLS)
    model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
    Price = model_OSL.params[0]
    Delta = model_OSL.params[1]
    return Price, Delta


def DeltafunkLSMISD(S0, K, T, N, mu, sigma, simnum, alpha):
    Stockprices = S0 + np.arange(-20, S0, 1)
    DeltaLSMISD = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        print(i)
        price, delta = twostage(simnum, sigma, mu, N, T, Stockprices[i], K, alpha)
        DeltaLSMISD[i] = delta
    return DeltaLSMISD

#Cashflow
def Cashftiden(simnum, sigma, mu, TimeToMaturity, S0, T, alpha, N0, N, K, Exerciseboundary):
    disc_vector = d(mu, N0, T)
    steps = N // N0  # steps
    exerciseboundary = Exerciseboundary / K

    Stock_price_paths = stock_priser_ISD_2(simnum, sigma, mu, TimeToMaturity, S0, T, alpha, N0)
    Stock_price_paths = np.array(Stock_price_paths) / K
    cashflow_matrix = np.zeros((simnum, (T * TimeToMaturity) + 1))

    for j in range(2, TimeToMaturity * T + 1, 1):
        exercise = Stock_price_paths[:, j] <= exerciseboundary[j * steps + (N0 - TimeToMaturity) * steps]
        Stock_price_paths[exercise, j + 1:] = 0
        Stock_price_paths[~exercise, j] = 0
    for j in range(TimeToMaturity * T, 0, -1):
        Cashflow = Stock_price_paths[:, j] > 0
        cashflow_matrix[Cashflow, j] = np.maximum(0, 1 - Stock_price_paths[Cashflow, j])
    cashflow_disc = np.matmul(cashflow_matrix[:, 2:(TimeToMaturity * T) + 1],
                              np.transpose(disc_vector[0:(TimeToMaturity * T - 1)]))
    return cashflow_disc, Stock_price_paths, cashflow_matrix, disc_vector


def regressionstep(simnum, sigma, mu, TimeToMaturity, S0, T, alpha, N0, N, K, Exerciseboundary):
    cashflow_disc, Stock_price_paths, cashflow_matrix, disc_vector = Cashftiden(simnum, sigma, mu, TimeToMaturity, S0,
                                                                                  T, alpha, N0, N, K, Exerciseboundary)
    X_regression = Stock_price_paths[:, 1]
    Y_regression = cashflow_disc
    Basis = laguerre_basis(X_regression)
    model = LinearRegression().fit(Basis, Y_regression)
    Continuation = model.predict(Basis)
    # Bestemmer value-funktionen paa tidspunkt 1 og diskonterer til tidspunkt 0
    Vt1 = np.maximum(Continuation, cashflow_matrix[:, 1])
    Vt1_disc = Vt1 * disc_vector[0]

    Y_OLS = K * Vt1_disc
    X = K * Stock_price_paths[:, 0]
    X_OLS = basepolynom(X, S0, 8)
    X_OLS = sm.add_constant(X_OLS)
    model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
    Price = model_OSL.params[0]
    Delta = model_OSL.params[1]
    return Price, Delta


# Hedging error
def DeltaLSMIDS(Exerciseboundary, StockpricesDelta, N0, ):
    Hedgingtime = hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0)
    HedgingerorIS = np.full(len(StockpricesDelta), np.nan)

    start = time.time()
    for j in range(800, 1000):
        print('stock path', j)
        Hedgingtimej = int(Hedgingtime[j])
        print('Hedgingtime', Hedgingtimej)
        StockpricesDeltaj = StockpricesDelta[j]
        price = np.zeros(Hedgingtimej)
        delta = np.zeros(Hedgingtimej)
        for i in range(0, Hedgingtimej):
            print(i)
            Price, Delta = regressionstep(100000, sigma, mu, N0 - i, StockpricesDeltaj[i], T, alpha, N0, N, K,
                                          Exerciseboundary)
            price[i] = Price
            delta[i] = Delta
        delta = np.clip(delta, -1, 0)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + delta[i - 1] * StockpricesDeltaj[i]
            Cash[i] = RP[i] - delta[i] * StockpricesDeltaj[i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + delta[Hedgingtimej - 1] * StockpricesDeltaj[Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerorISD[j] = dischedgerror
    end = time.time()
    print(end - start)
    return HedgingerorISD


def deltaLSMISD2(Exerciseboundary, simnum, sigma, mu, N, S0, T, alpha, K):
    stockprices = stockpriserISD(simnum, sigma, mu, N, S0, T, alpha)
    disc_vector = d(mu, N, T)
    coef = np.zeros((N, 9))
    for i in range(0, N):
        Stockpricesoptim = np.array(stockprices)
        cashflow_matrix = np.zeros((simnum, (T * N) + 1))
        for j in range(i + 2, N * T + 1, 1):
            exercise = Stockpricesoptim[:, j] <= Exerciseboundary[j]
            Stockpricesoptim[exercise, j + 1:] = 0
            Stockpricesoptim[~exercise, j] = 0

        for j in range(N * T, i, -1):
            Cashflow = Stockpricesoptim[:, j] > 0
            cashflow_matrix[Cashflow, j] = np.maximum(0, K - Stockpricesoptim[Cashflow, j])

        cashflow_disc = np.matmul(cashflow_matrix[:, i + 2:(N * T) + 1], np.transpose(disc_vector[0:(N * T) - (i + 1)]))

        X1 = stockprices[:, i + 1]
        Basis = laguerre_basis(X1)
        Y_regression = cashflow_disc
        model = LinearRegression().fit(Basis, Y_regression)
        Continuation = model.predict(Basis)

        # Bestemmer value-funktionen på tidspunkt 1 og diskonterer til tidspunkt 0
        Vt1 = np.maximum(Continuation, cashflow_matrix[:, 1 + i])
        Vt1_disc = Vt1 * disc_vector[0]

        # Greeks igen
        Y_OLS = Vt1_disc
        X = stockprices[:, i]
        X_OLS = basepolynom(X, S0, 8)
        X_OLS = sm.add_constant(X_OLS)
        model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
        coef[i, :] = model_OSL.params
    return coef


# Deltafunktionen LsmIsd
def deltafunktion2(Stockprice, S0, coef, Hedgingtime):
    delta = np.full(len(Stockprice), np.nan)
    for i in range(Hedgingtime):
        delta[i] = coef[i, 1] + 2 * coef[i, 2] * (Stockprice[i] - S0) + 3 * coef[i, 3] * (Stockprice[i] - S0) ** 2 + 4 * \
                   coef[i, 4] * (Stockprice[i] - S0) ** 3 + 5 * coef[i, 5] * (Stockprice[i] - S0) ** 4 + 6 * coef[
                       i, 6] * (Stockprice[i] - S0) ** 5 + 7 * coef[i, 7] * (Stockprice[i] - S0) ** 6 + 8 * coef[
                       i, 8] * (Stockprice[i] - S0) ** 7
    Delta = np.clip(delta, -1, 0)
    return Delta


# HEDGING ERROR - LSM ISD
def DeltafunktiongenLSMISD(Exerciseboundary, StockpricesDelta, N2, N0, simnum, sigma, mu, N, S0, T, alpha, K):
    coef = deltaLSMISD2(Exerciseboundary, simnum, sigma, mu, N, S0, T, alpha, K)
    Hedgingtime = hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0)
    HedgingerorISD2 = np.full(len(StockpricesDelta), np.nan)
    for j in range(len(StockpricesDelta)):
        Hedgingtimej = int(Hedgingtime[j])
        StockpricesDeltaj = StockpricesDelta[j]
        Delta = deltafunktion2(StockpricesDeltaj, S0, coef, Hedgingtimej)
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
        HedgingerorISD2[j] = dischedgerror
    return HedgingerorISD2


# Deltafunk. metoden
def DeltafunkLSMISD2(S0, K, T, N, r, sigma):
    Stockprices = S0 + np.arange(-20, S0, 1)
    DeltafunkISD2 = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        DeltafunkISD2[i] = coef[0, 1] + 2 * coef[0, 2] * (Stockprices[i] - S0) + 3 * coef[0, 3] * (
                Stockprices[i] - S0) ** 2 + 4 * coef[0, 4] * (Stockprices[i] - S0) ** 3 + 5 * coef[0, 5] * (
                                   Stockprices[i] - S0) ** 4 + 6 * coef[0, 6] * (Stockprices[i] - S0) ** 5 + 7 * coef[
                               0, 7] * (Stockprices[i] - S0) ** 6 + 8 * coef[0, 8] * (Stockprices[i] - S0) ** 7
    DeltafunkISD2 = np.clip(DeltafunkISD2, -1, 0)
    return DeltafunkISD2


def inputs2stage():
    option_priser, exerciseboundary = lsm(simnum, sigma, mu, N, T, K - 4, K)
    Stock_price_paths = sim_stock_prices(simnum, sigma, mu, N, S0, T, alpha)
    Stock_price_paths = np.array(Stock_price_paths) / K
    return Stock_price_paths, exerciseboundary


def hedgetime(exerciseboundary, N2, N0, sigma, mu, T, S0, K):
    exercisesteps = N2 // N0
    optionalive = np.zeros(N0 - 1)
    Exerciseboundary = exerciseboundary * K
    Stockprice = sim_stock_prices2(1, sigma, mu, N0, T, S0)  # fra GBMbase.py
    for i in range(1, N0):
        if Stockprice[0, i] > Exerciseboundary[i * exercisesteps]:
            optionalive[i - 1] = 1
        else:
            break
    HedgeTime = int(sum(optionalive)) + 1
    return HedgeTime, Exerciseboundary, Stockprice


# Har heller ikke brugt følgende her.
def DeltaLSM_ISD(HedgeTime, Stockprice, exerciseboundary, simnum, sigma, mu, T, alpha, N0, N, K):
    price = np.zeros(HedgeTime)
    delta = np.zeros(HedgeTime)
    for i in range(0, HedgeTime):
        Price, Delta = regressionstep(simnum, sigma, mu, N0 - i, Stockprice[0, i], T, alpha, N0, N, K, exerciseboundary)
        price[i] = Price
        delta[i] = Delta
    Cash = np.zeros(HedgeTime)
    RP = np.zeros(HedgeTime + 1)
    RP[0] = price[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = price[0] - delta[0] * Stockprice[0, 0]
    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - delta[i] * Stockprice[0, i]
    RP[HedgeTime] = Cash[HedgeTime - 1] * disc + delta[HedgeTime - 1] * Stockprice[0, HedgeTime]
    price = np.append(price, np.maximum(0, K - Stockprice[0, HedgeTime]))
    hedgerror = RP[HedgeTime] - price[HedgeTime]
    timetomaturity = (N0 - HedgeTime) / N0
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    Cash = np.zeros(Hedgingtimej)
    RP = np.zeros(Hedgingtimej + 1)
    RP[0] = price[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = price[0] - delta[0] * Stockprice[0, 0]

    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - delta[i] * Stockprice[0, i]
    RP[HedgeTime] = Cash[Hedgingtimej - 1] * disc + delta[Hedgingtimej - 1] * Stockprice[0, HedgeTime]
    price = np.append(price, np.maximum(0, K - Stockprice[0, HedgeTime]))
    hedgerror = RP[HedgeTime] - price[HedgeTime]
    timetomaturity = (N0 - HedgeTime) / N0
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    HedgingerorISD[j] = dischedgerror
    return dischedgerror
