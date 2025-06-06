import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

simnum = 5 #simulates paths
sigma = 0.2
mu = 0.06
N = 50
T = 1 #Maturity time in years
S0 = 36
K = 44
alpha=0.5

# med antithetic paths
def sim_stock_prices(simnum, sigma, mu, N, T, S0):
    dt = 1/N
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = S0
    for i in range(0,simnum,2):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
            Stock_price_paths[i+1, j] = Stock_price_paths[i+1, j - 1] * np.exp((mu - sigma ** 2 / 2) * dt - sigma *z*np.sqrt(dt))
    return Stock_price_paths

#uden Antithetic paths
def sim_stock_prices2(simnum, sigma, mu, N, T, S0):
    dt = 1/N
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = S0
    for i in range(0,simnum,1):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
    return Stock_price_paths

#Simulation af aktiepriser
def sim_stock_pricesflex(simnum, sigma, mu, N, S0,T, N0):
    dt=1/N0
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = S0
    for i in range(0,simnum,2):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
            Stock_price_paths[i+1, j] = Stock_price_paths[i+1, j - 1] * np.exp((mu - sigma ** 2 / 2) * dt - sigma *z*np.sqrt(dt))
    return Stock_price_paths