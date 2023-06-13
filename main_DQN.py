import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# device selection
DEVICE = torch.device("cpu")
THR_SHARPE = 2
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed
seed_everything(0)


class Dqn(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_layer, n_nodes):
        super().__init__()
        self.n_layer = n_layer
        self.n_outputs = n_outputs

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=n_nodes),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=n_nodes, out_features=n_nodes),
                    nn.ReLU()
                )
                for _ in range(n_layer - 1)]
        )

        self.output_layer = nn.Linear(in_features=n_nodes, out_features=n_outputs)

    def forward(self, x):
        input_layer_out = self.input_layer(x.type(torch.float32))

        hidden_layers_out = self.hidden_layers[0](input_layer_out)
        for i in range(0, self.n_layer - 1):
            hidden_layers_out = self.hidden_layers[i](hidden_layers_out)

        output = self.output_layer(hidden_layers_out)

        return output


class Agent:
    def __init__(self, n_asset, input_size):
        self.n_asset = n_asset
        self.action_space = 40  # 0 BUY, 1 SELL
        self.nLayers = 10
        self.nNodes = 32

        self.gamma = 0.7  # discount factor
        self.epsilon = 1  # action epsilon-greedy --> then multiplied by 0.999

        self.epsilon_avoid_0 = 1e-9  # epsilon in order to avoid division by 0
        self.DQN_net = Dqn(n_inputs=input_size, n_outputs=self.action_space, n_layer=self.nLayers,
                           n_nodes=self.nNodes).to(DEVICE)

    def act(self, state):
        # with epsilon probability choose random
        decay_factor = 0.999
        self.epsilon *= decay_factor  # epsilon greedy action selection decrease over time
        if random.random() < self.epsilon:
            # random choice an action to perform
            # weights = np.random.uniform(-5, 5, size=self.n_asset)
            weights = np.random.normal(size=self.n_asset)

            weights_sum = np.sum(weights) + self.epsilon_avoid_0
            weights /= weights_sum
            weights = torch.tensor(weights, device=DEVICE).view(1, self.n_asset)
            best_actions = torch.randint(low=0, high=2, size=(20,), device=DEVICE)

        # otherwise perform the best actions
        else:
            with torch.no_grad():
                # select q(s,a) from network
                linear_actions = self.DQN_net(state)
            linear_actions = linear_actions.reshape(int(linear_actions.shape[0] / 2), 2)
            # select best actions from network output
            best_actions = torch.argmax(linear_actions, dim=1)

            weights = torch.empty((1, self.n_asset), device=DEVICE)

            # iterate over the actions
            for index, value in enumerate(best_actions):
                if value == 0:  # BUY
                    weights[0][index] = linear_actions[index][value]
                else:  # SELL
                    weights[0][index] = -linear_actions[index][value]

            # normalize the weights to sum to 1
            weights_sum = torch.sum(weights) + self.epsilon_avoid_0
            weights /= weights_sum

        return weights, best_actions


@dataclass
class State:
    cov: torch.Tensor
    last_allocation: torch.Tensor
    portfolio_value: float
    data: torch.Tensor
    es: torch.Tensor

    def __len__(self):
        return self.cov.shape[0] * self.cov.shape[1] + self.last_allocation.shape[1] + 1 + \
            self.data.shape[0] * self.data.shape[1] + self.data.shape[1]

    def flatten(self):
        return torch.cat(
            (
                self.cov.flatten(),
                self.last_allocation.flatten(),
                torch.tensor([self.portfolio_value], device=DEVICE).flatten(),
                self.data.flatten(),
                self.es.flatten()
            )
        )

    @classmethod
    def from_flatten(cls, flat_state: torch.Tensor, n_stock: int, lag: int):
        cov = flat_state[:n_stock * 2].reshape((n_stock, n_stock))
        last_allocation = flat_state[n_stock * 2: n_stock * 2 + n_stock]
        portfolio_value = flat_state[n_stock * 2 + n_stock]
        data = flat_state[n_stock * 2 + n_stock + 1:-1].reshape((lag, n_stock))
        es = flat_state[-1:0]
        s = cls(cov, last_allocation, portfolio_value, data, es)
        return s


class Environment:
    def __init__(self, data_path, money):
        self.data_path = data_path
        self.money = money
        self.money_reset = money
        self.dates = None
        self.real_data = self.load_data()
        self.data = None

    def generate_data(self, data):
        def generate_stock_data(actual_hist):
            # Parameters
            # drift coefficient
            mu = actual_hist.pct_change()[1:].mean()
            # number of steps
            n = 255
            # number of sims
            M = 10
            # initial stock price
            S0 = actual_hist.iloc[-1]
            # volatility
            sigma = actual_hist.std() / 100

            # calc each time step
            dt = 2. / (n - 1)
            # simulation using numpy arrays
            St = np.exp(
                (mu - sigma ** 2 / 2) * dt
                + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
            )
            # include array of 1's
            St = np.vstack([np.ones(M), St])
            # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
            St = S0 * St.cumprod(axis=0)
            return St[:, random.choices(range(St.shape[1]))[0]]

        return pd.DataFrame({col: generate_stock_data(data[col]) for col in data.columns})

    def get_starting_state(self, lag, confidence_level):
        idx = random.randint(255, len(self.data))
        data = self.data.iloc[idx-255:idx]
        self.data = self.generate_data(data)
        initial_alloc = torch.zeros((1, 20), device=DEVICE)
        state = self.get_new_state(t=lag+1, lag=lag, last_alloc=initial_alloc, portfolio_val=self.money_reset, confidence_level=confidence_level)
        return state

    def load_data(self):
        # read aggregated real_data like Dataframe
        # and then convert it to torch.tensor
        # with % between t+1 and t values for
        # each column
        data = pd.read_csv(self.data_path)
        self.dates = data['Date'].copy(deep=True)
        data = data.drop(columns=['Date'])
        data = torch.tensor(data.values, device=DEVICE)
        # A vector of zeros with shape 1, N_STOCK
        data_copy = torch.zeros((1, data.shape[1]), device=DEVICE)
        data_copy = torch.cat((data_copy, data[:-1]))
        res = (((data - data_copy) / data_copy) * 100)[1:, :]
        return res

    def get_new_state(self, t: int, lag: int, last_alloc, portfolio_val: float, confidence_level):
        data_lag = self.data[t - lag:t, :]

        # compute covariance matrix for lag prices ago till t
        cov_matrix = torch.cov(data_lag.t())

        # compute VaR
        sorted_returns, _ = torch.sort(data_lag, dim=0)
        var_position = int(torch.floor((1 - torch.tensor(confidence_level, device=DEVICE)) * data_lag.shape[1]))
        var = -sorted_returns[:, var_position]

        # Calculate ES for each asset
        es = -sorted_returns[:var_position, :].mean(dim=0)

        return State(cov_matrix, last_alloc, portfolio_val, data_lag, es)

    def get_reward(self, weights, next_day_reward, lag, es):
        # get percentage change of t+1
        next_day = self.real_data[next_day_reward, :]

        # reward = 0
        risk_free_rate = 5.12  # USA Department of Treasury (3 months interest rate)

        portolio_returns = torch.mul(next_day.to(torch.float32), weights.flatten())
        portfolio_total_return = portolio_returns.sum()
        self.money += self.money * portfolio_total_return

        # rets = self.real_data[next_day_reward - lag:next_day_reward, :]
        # rets_mean = torch.mean(portolio_returns, dim=0)
        # rets_cov = torch.cov(portolio_returns)
        P_std = portolio_returns.std()
        # P_dev_std = torch.sqrt(
        #     torch.mm(weights.to(torch.float32), torch.multiply(rets_cov.to(torch.float32), weights.t().to(torch.float32))))
        P_sharpe = (portfolio_total_return - risk_free_rate) / P_std
        # Good Sharpe ratio
        reward = P_sharpe.item()

        done = 1 if self.money < 0 else 0
        # if done == 1:
        #     reward += -100

        return reward, done


def DQNUpdate(neural_net, memory_buffer, optimizer, agent, device, state_dim: int, BATCH_SIZE):
    """
    Main update rule for the DQN process. Extract real_data from the memory buffer and update
    the newtwork computing the gradient.

    """
    criterion = nn.MSELoss()

    if len(memory_buffer) < BATCH_SIZE: return

    n_asset = agent.n_asset
    gamma = agent.gamma
    # memory_buffer = memory_buffer[torch.randperm(memory_buffer.shape[0])][:BATCH_SIZE, :]
    memory_buffer = memory_buffer[-BATCH_SIZE:, :]

    state = memory_buffer[:, :state_dim]
    action = memory_buffer[:, state_dim:state_dim + n_asset]
    next_state = memory_buffer[:, state_dim + n_asset:(state_dim * 2) + n_asset]
    reward = memory_buffer[:, (state_dim * 2) + n_asset].reshape(BATCH_SIZE, 1)
    done = memory_buffer[:, (state_dim * 2) + n_asset + 1].reshape(BATCH_SIZE, 1)

    # MAYBE WITH GRAD
    target = neural_net(state)
    target = target.reshape(target.shape[0], int(target.shape[1] / 2), 2)

    # if done --> to_update = reward
    output_net = neural_net(next_state)
    output_net = output_net.reshape(output_net.shape[0], int(output_net.shape[1] / 2), 2)
    to_update = reward + torch.max(output_net, 2).values * gamma * (1 - done)

    for row_n in range(target.shape[0]):
        for column_n in range(target.shape[1]):
            target[row_n, column_n, action[row_n, column_n].to(torch.int32)] = to_update[row_n][column_n]

    predicted = neural_net(state)
    predicted = predicted.reshape(predicted.shape[0], int(predicted.shape[1] / 2), 2)

    loss = criterion(predicted.to(torch.float32), target.to(torch.float32))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    # Expected Shortfall threshold
    confidence_level = 0.95

    # Environment() parameters
    train_data_path = 'TRAIN.csv'
    lag = 30
    money = 10_000
    n_asset = 20

    # instantiate environment
    env = Environment(data_path=train_data_path, money=money)
    data = env.real_data

    state = env.get_starting_state(lag, confidence_level)
    STATE_DIM = len(state)
    # instantiate agent
    agent = Agent(n_asset=n_asset, input_size=STATE_DIM)

    # index to split train-test
    percentage_train = 80
    index_train = int((percentage_train / 100) * data.shape[0])

    # set limit memory
    BATCH_SIZE = 16

    # instantiate the optimizer
    optimizer_learning_rate = 0.0001
    optimizer = torch.optim.Adam(agent.DQN_net.parameters(), lr=optimizer_learning_rate)

    # memory buffer and its dimensionality
    # ATTRIBUTE
    # state,            action,      next_state,        reward       done
    # DIMENSION
    # LEN STATE         n_asset      LEN STATE     1             1
    memory_buffer = torch.empty((0, (STATE_DIM * 2) + 2 + n_asset), device=DEVICE)

    # iterate over all t in train part
    episode = 1
    reward_evolution = []
    n_episodes = 100
    for next_t in range(n_episodes):
        env.money = env.money_reset

        while True:
            weights, best_actions = agent.act(state.flatten())
            reward, done = env.get_reward(weights=weights, next_day_reward=next_t, lag=lag, es=state.es)
            if done:
                break
            next_state = env.get_new_state(t=next_t, lag=lag, last_alloc=weights, portfolio_val=env.money,
                                           confidence_level=confidence_level)
            reward_evolution.append(reward)
            step = torch.cat(
                (
                    state.flatten(),
                    best_actions.flatten(),
                    next_state.flatten(),
                    torch.tensor(reward, device=DEVICE).flatten(),
                    torch.tensor(done, device=DEVICE).flatten())
            )
            memory_buffer = torch.cat((memory_buffer, step.view(1, step.shape[0])), dim=0)
            state = next_state

            # TRAINING LAUNCH
            if len(memory_buffer) >= BATCH_SIZE:
                # for _ in range(10):
                DQNUpdate(neural_net=agent.DQN_net, memory_buffer=memory_buffer,
                          optimizer=optimizer, agent=agent, device=DEVICE, state_dim=STATE_DIM, BATCH_SIZE=BATCH_SIZE)
            print(f"EPISODE {episode}, PORTFOLIO: {env.money}, EPS: {agent.epsilon}")
            episode += 1

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(reward_evolution)), reward_evolution, color='green', label='reward train')
    plt.axhline(y=THR_SHARPE, color='r', linestyle='-')
    plt.title('Reward evolution')
    plt.legend()
    plt.show()

    # save the net model
    torch.save(agent.DQN_net, f'DQN_net.pth')


def test():
    # Expected Shortfall threshold
    confidence_level = 0.95

    # Environment() parameters
    test_data_path = 'TEST.csv'
    lag = 30
    money = 10_000
    n_asset = 20

    # instantiate environment
    env = Environment(data_path=test_data_path, money=money)
    data = env.real_data
    dates = env.dates

    initial_alloc = torch.zeros((1, 20), device=DEVICE)
    state = env.get_new_state(t=lag + 1, lag=lag, last_alloc=initial_alloc, portfolio_val=env.money,
                              confidence_level=confidence_level)
    STATE_DIM = len(state)
    DQN_net = torch.load('DQN_net.pth')
    DQN_net.eval()
    # instantiate agent
    agent = Agent(n_asset=n_asset, input_size=STATE_DIM)

    agent.epsilon = 0
    agent.DQN_net = DQN_net

    episode = 1
    reward_evolution = []
    wallet_evolution = [money]
    weights_evolution = np.empty(shape=(1, n_asset))
    for next_t in range(data.shape[0]):
        weights, best_actions = agent.act(state.flatten())
        weights_evolution = np.vstack((weights_evolution, weights.detach().cpu().numpy()))
        reward, done = env.get_reward(weights=weights, next_day_reward=next_t, lag=lag, es=state.es)
        next_state = env.get_new_state(t=next_t, lag=lag, last_alloc=weights, portfolio_val=env.money,
                                       confidence_level=confidence_level)
        state = next_state
        print(f"EPISODE {episode},PORTFOLIO: {env.money}")
        wallet_evolution.append(env.money.item())
        reward_evolution.append(reward)
        if done == 1:
            break
        episode += 1

    # Plot Money Evolution
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame({"Dates": dates.values[:len(wallet_evolution)], "money": wallet_evolution})
    df.set_index('Dates', inplace=True)
    ax = df.plot.area(figsize=(12, 6), color='green', stacked=False)
    ax.collections[0].set_facecolor('lightblue')
    ax.collections[0].set_alpha(0.3)
    plt.xticks(rotation=45)
    plt.title('Wallet evolution')
    plt.axhline(y=money, color='r', linestyle='-')
    plt.tight_layout()
    plt.legend()
    plt.savefig('money.png', dpi=200)
    plt.show()

    # Plot Reward Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(reward_evolution)), reward_evolution, color='red',
             label='reward test')
    plt.title('Reward evolution')
    plt.legend()
    plt.savefig('reward.png', dpi=200)
    plt.show()
    print(f"CUMULATIVE REWARD: {sum(reward_evolution)}")
    print(f"AVG REWARD: {(1 / len(reward_evolution)) * sum(reward_evolution)}")
    print(f"ROI: {round((wallet_evolution[-1] - money) / money * 100, 2)}%")


if __name__ == "__main__":
    # train()
    test()