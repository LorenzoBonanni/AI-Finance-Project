import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Dqn(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_layer, n_nodes):
        super().__init__()
        self.n_layer = n_layer

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
    def __init__(self, n_asset, device):
        self.n_asset = n_asset
        self.action_space = 2  # 0 BUY, 1 SELL
        self.nLayers = 2
        self.nNodes = 32
        self.device = device

        self.alpha = 0.5  # learning rate
        self.gamma = 0.95  # discount factor
        self.epsilon = 0.1  # action epsilon-greedy

        self.epsilon_avoid_0 = 1e-9  # epsilon in order to avoid division by 0

        self.DQN_net = Dqn(n_inputs=self.n_asset, n_outputs=self.action_space, n_layer=self.nLayers,
                           n_nodes=self.nNodes).to(self.device)

    def act(self, state):
        # with epsilon probability choose random
        if random.random() < self.epsilon:
            # random choice an action to perform
            weights = np.random.uniform(-5, 5, size=self.n_asset)

            weights_sum = np.sum(weights) + self.epsilon_avoid_0
            weights /= weights_sum
            weights = torch.tensor(weights, device=self.device).view(1,self.n_asset)
            best_actions = torch.randint(low=0, high=2, size=(20,), device=self.device)

        # otherwise perform the best actions
        else:
            # select q(s,a) from network
            linear_actions = self.DQN_net(state)
            # select best actions from network output
            best_actions = torch.argmax(linear_actions, dim=1)

            weights = torch.empty((1, self.n_asset), device=self.device)

            # iterate over the actions
            # TODO optimize
            for index, value in enumerate(best_actions):
                if value == 0:  # BUY
                    weights[0][index] = linear_actions[index][value]
                else:  # SELL
                    weights[0][index] = -linear_actions[index][value]

            # normalize the weights to sum to 1
            weights_sum = torch.sum(weights) + self.epsilon_avoid_0
            weights /= weights_sum

        return weights, best_actions


class Environment:
    def __init__(self, path_assets, device, money):
        self.path_assets = path_assets
        self.money = money
        self.device = device
        self.data = self.load_data()

    def load_data(self):
        # read aggregated data like Dataframe
        # and then convert it to torch.tensor
        # with % between t+1 and t values for
        # each column
        data = pd.read_csv(self.path_assets)
        data = data.drop(columns=['Date'])
        data = torch.tensor(data.values, device=self.device)
        # A vector of zeros with shape 1, N_STOCK
        data_copy = torch.zeros((1, data.shape[1]), device=self.device)
        data_copy = torch.cat((data_copy, data[:-1]))
        res = (((data - data_copy) / data_copy) * 100)[1:, :]
        return res

    def get_new_state(self, t, lag):
        # compute covariance matrix
        # for lag prices ago till t
        data_lag = self.data[t - lag:t, :]
        cov_matrix = torch.cov(data_lag.t())
        return cov_matrix

    def get_reward(self, weights, next_day_reward):
        # get percentage change of t+1
        next_day = self.data[next_day_reward, :]
        # compute the allocation in real money to each asset
        allocation_money = weights * self.money

        reward = 0

        # iterate over each percentage
        for index, value in enumerate(next_day):

            variation_money_allocation =  (value / 100) * allocation_money[0][index]


            # SELL
            if weights[0][index] < 0:
                # if percentage < 0, SELL is best
                if value < 0:
                    reward += -1
                    self.money += variation_money_allocation * (-1)
                # percentage >= 0, SELL is NOT good
                else:
                    reward += 1
                    self.money += variation_money_allocation * (-1)
            # BUY
            else:
                # if percentage < 0, BUY is NOT good
                if value < 0:
                    reward += 1
                    self.money += variation_money_allocation
                # percentage >= 0, BUY is best
                else:
                    reward += -1
                    self.money += variation_money_allocation
        return reward


def DQNUpdate(neural_net, memory_buffer, optimizer, agent, device, batch_size=32):
    """
    Main update rule for the DQN process. Extract data from the memory buffer and update
    the newtwork computing the gradient.

    """
    criterion = nn.MSELoss()

    if len(memory_buffer) < batch_size: return

    n_asset = agent.n_asset
    gamma = agent.gamma
    action_space = agent.action_space

    memory_buffer = memory_buffer[torch.randperm(memory_buffer.shape[0])][:200, :]

    state_dim = n_asset * n_asset
    number_step_buffer = memory_buffer.shape[0]

    state = memory_buffer[:, :state_dim].reshape(number_step_buffer, n_asset, n_asset)
    action = memory_buffer[:, state_dim:state_dim + n_asset].reshape(number_step_buffer, n_asset)
    next_state = memory_buffer[:, state_dim + n_asset:(state_dim * 2) + n_asset].reshape(number_step_buffer, n_asset,
                                                                                         n_asset)
    reward = memory_buffer[:, (state_dim * 2) + n_asset].reshape(number_step_buffer, 1)

    target = neural_net(state)
    # target_step = torch.empty([0, action_space], device=device, requires_grad=True)

    target_copy = torch.clone(target) # (32,20,2)

    with torch.no_grad():
        to_update = reward + torch.max(neural_net(next_state),2).values * gamma

        for row_n in range(target_copy.shape[0]):
            for column_n in range(target_copy.shape[1]):
                target_copy[row_n,column_n,action[row_n,column_n].to(torch.int64)] = to_update[row_n][column_n]


    predicted = neural_net(state)

    objective_done = criterion(predicted, target_copy)

    optimizer.zero_grad()
    objective_done.backward()
    optimizer.step()


def main():
    # device selection
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Environment() parameters
    path_assets = 'AGGREGATED_DATA.csv'
    lag = 10
    money = 1000
    # Agent() parameters
    n_asset = 20

    # instantiate environment
    env = Environment(path_assets=path_assets, device=device, money=money)
    data = env.data

    # instantiate agent
    agent = Agent(n_asset=n_asset, device=device)

    # index to split train-test
    percentage_train = 80
    index_train = int((percentage_train / 100) * data.shape[0])

    # set limit memory
    BATCH_SIZE = 16

    # instantiate the optimizer
    optimizer = torch.optim.Adam(agent.DQN_net.parameters(), lr=0.001)

    # memory buffer and its dimensionality
    # ATTRIBUTE
    # state,            action,      next_state,        reward
    # DIMENSION
    # n_asset*n_asset   n_asset     n_asset*n_asset     1
    memory_buffer = torch.empty((0, ((n_asset * n_asset) * 2) + n_asset + 1), device=device)

    # iterate over all t in train part
    state = env.get_new_state(t=lag + 1, lag=lag)
    episode = 1
    averaged_rewards = []
    for next_t in range(lag + 2, index_train):
        weights, best_actions = agent.act(state)
        reward = env.get_reward(weights=weights, next_day_reward=next_t)
        next_state = env.get_new_state(t=lag, lag=lag)
        step = torch.cat(
            (state.flatten(), best_actions.flatten(), next_state.flatten(),
             torch.tensor(reward, device=device).flatten()))
        memory_buffer = torch.cat((memory_buffer, step.view(1, step.shape[0])), dim=0)
        next_state = state

        # TRAINING LAUNCH
        if len(memory_buffer) >= BATCH_SIZE:
            for _ in range(10):
                DQNUpdate(neural_net=agent.DQN_net, memory_buffer=memory_buffer,
                          optimizer=optimizer, agent=agent, device=device)
        averaged_rewards.append(reward)
        print(f"EPISODE {episode}, AVEREGED_REWARD {np.mean(averaged_rewards)}, PORTFOLIO: {env.money}")
        episode += 1

if __name__ == "__main__":
    main()
