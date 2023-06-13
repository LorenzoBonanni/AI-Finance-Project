import random

import numpy as np
import pandas as pd


# plt.figure(figsize=(12, 6))
# plt.plot(range(len(St)), St, 'b')
# plt.plot(range(len(actual_hist)), actual_hist, 'r', label="Real Data")
# plt.xlabel("Days $(t)$")
# plt.ylabel("Stock Price $(S_t)$")
# plt.title(
#     "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(
#         S0, mu, sigma)
# )
#
# legend_elements = [Patch(facecolor='b', edgecolor='k', label='Brownian Motion'),
#                    Patch(facecolor='r', edgecolor='k', label='Real Data"')]
# plt.gca().legend(handles=legend_elements, loc='upper left')
# plt.tight_layout()
# plt.savefig('gbm.png', dpi=200)
# plt.show()

# def generate_data(real_data):
#     def generate_stock_data(actual_hist):
#         # Parameters
#         # drift coefficient
#         mu = actual_hist.pct_change()[1:].mean()
#         # number of steps
#         n = 255
#         # number of sims
#         M = 10
#         # initial stock price
#         S0 = actual_hist.iloc[-1]
#         # volatility
#         sigma = actual_hist.std() / 100
#
#         # calc each time step
#         dt = 2. / (n - 1)
#         # simulation using numpy arrays
#         St = np.exp(
#             (mu - sigma ** 2 / 2) * dt
#             + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
#         )
#         # include array of 1's
#         St = np.vstack([np.ones(M), St])
#         # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
#         St = S0 * St.cumprod(axis=0)
#         return St[:, random.choices(range(St.shape[1]))[0]]
#
#     return pd.DataFrame({col: generate_stock_data(real_data[col]) for col in real_data.columns})


train_data = pd.read_csv('TRAIN.csv').iloc[-255:]
train_data.set_index('Date', inplace=True)
df = generate_data(train_data)

print(df)
