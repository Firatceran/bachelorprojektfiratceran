import statsmodels.api as sm
from LsmLongstaffSchwartz import *
import seaborn as sns


def ISD(simnum, S0, alpha):
    U = np.zeros(simnum)
    U_2 = np.random.uniform(0, 1, (simnum // 2,))
    for i in range(simnum // 2):
        U[2 * i] = U_2[i]
        U[2 * i + 1] = U_2[i]
    K_isd = 2 * np.sin(np.arcsin(2 * U - 1) / 3)
    X = S0 + alpha * K_isd
    return X


def stockpriserISD(simnum, sigma, mu, N, S0, T, alpha):
    dt = 1 / N
    Stock_price_paths = np.zeros((simnum, (T * N) + 1))
    # Generate stock prices
    Stock_price_paths[:, 0] = ISD(simnum, S0, alpha)
    for i in range(0, simnum, 2):
        for j in range(1, (T * N) + 1):
            z = np.random.normal(loc=0, scale=1)
            Stock_price_paths[i, j] = Stock_price_paths[i, j - 1] * np.exp(
                (mu - sigma ** 2 / 2) * dt + sigma * z * np.sqrt(dt))
            Stock_price_paths[i + 1, j] = Stock_price_paths[i + 1, j - 1] * np.exp(
                (mu - sigma ** 2 / 2) * dt - sigma * z * np.sqrt(dt))
    return Stock_price_paths


def basepolynom(X, x0, degree):
    basis = [(X - x0) ** j for j in range(1, degree + 1)]
    return np.column_stack(basis)


def Olsgreek(simnum, sigma, mu, N, T, S0, K, alpha):
    disc_vector = d(mu, N, T)
    option_priser, exerciseboundary = lsm(simnum, sigma, mu, N, T, K - 4, K)
    Stock_price_paths = stockpriserISD(simnum, sigma, mu, N, S0, T, alpha)
    X = Stock_price_paths[:, 0]
    Stock_price_paths = np.array(Stock_price_paths) / K
    cashflow_matrix = np.zeros((simnum, (T * N) + 1))

    for j in range(1, N * T + 1, 1):
        exercise = Stock_price_paths[:, j] <= exerciseboundary[j]
        Stock_price_paths[exercise, j + 1:] = 0
        Stock_price_paths[~exercise, j] = 0
    for j in range(N * T, 0, -1):
        Cashflow = Stock_price_paths[:, j] > 0
        cashflow_matrix[Cashflow, j] = np.maximum(0, 1 - Stock_price_paths[Cashflow, j])

    cashflow_disc = np.matmul(K * cashflow_matrix[:, 1:(N * T) + 1], np.transpose(disc_vector))
    Y_OLS = cashflow_disc
    X_OLS = basepolynoM(X, S0, 8)
    X_OLS = sm.add_constant(X_OLS)
    model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
    Price = model_OSL.params[0]
    Delta = model_OSL.params[1]
    return Price, Delta, Y_OLS, X


def tabel2(simnum):
    Strikeprise = np.array((36, 40, 44))
    PriceLSMISD1 = np.zeros((20, 3))
    DeltaLSMISD1 = np.zeros((20, 3))
    for j in range(3):
        print(j)
        for i in range(20):
            Price, Delta = Olsgreek(100000, 0.2, 0.06, 50, 1, 40, Strikeprise[j], 25)
            PriceLSMISD1[i, j] = Price
            DeltaLSMISD1[i, j] = Delta
    column_averagesPrice = np.mean(PriceLSMISD1, axis=0)
    column_std_devPrice = np.std(PriceLSMISD1, axis=0, ddof=1)
    column_averagesDelta = np.mean(DeltaLSMISD1, axis=0)
    column_std_devDelta = np.std(DeltaLSMISD1, axis=0, ddof=1)
    return


params = Olsgreek(100000, 0.2, 0.06, 50, 1, 36, 40, 5)
print(params)

simnum = 100000
sigma = 0.2
mu = 0.06
N = 50
T = 1
S0 = 40
K = 40
alpha = 0.5
