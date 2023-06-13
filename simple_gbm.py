import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_data = pd.read_csv('TRAIN.csv')
actual_hist = train_data['Adj Close_AAPL'].iloc[-255:]

# Generate MC simulation
# How many steps (days): stepsize = 1day
n_t = len(actual_hist)
print("Number of Days", n_t)
n_mc = 10000

# Initialize array S(t) -- container for the MC simualtions
St = pd.DataFrame(0., index=actual_hist.index, columns=list(range(1, n_mc + 1)))
St.iloc[0] = actual_hist.iloc[0]

# Annualized Volatility
sigma = actual_hist.std() / 100

# drift (buisiness cycle, long-term growth assumption)
mu = actual_hist.pct_change()[1:].mean()

# Step Size (what's 1 day in unif of year)
dt = 2. / (n_t - 1)
print("Daily Vol:", sigma * np.sqrt(dt))

# MC simulations with numpy's random number generator

for i in range(1, n_t):
    dS_2_S = mu * dt + sigma * np.sqrt(dt) * np.random.randn(n_mc)
    St.iloc[i] = St.iloc[i - 1] + St.iloc[i - 1] * dS_2_S

# Visualize and see if we did everything sensible
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)

for i in np.random.choice(St.columns.values, size=20):
    ax1.plot(St[i], 'b', lw=0.5)

ax1.plot(actual_hist, 'r', lw=1)  # SHow the actual to compare with MC paths
fig.show()
