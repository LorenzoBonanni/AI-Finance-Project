import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

def data_split(data, x_number, y_number):

    batch = torch.empty((0), device=device)

    for t in range(data.shape[0]  - (x_number+y_number)):
        train = torch.tensor(data[t:t+x_number], device=device, dtype=torch.float32).view(1,x_number)
        test = torch.tensor(data[t+x_number:t+x_number+y_number], device=device, dtype=torch.float32).view(1,y_number)

        batch_row = torch.cat((train,test), dim=1)

        batch = torch.cat((batch, batch_row), dim=0)

    return batch



if __name__ == "__main__":

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    stock = 'AAPL'

    data = pd.read_csv(f'./data_feed/{stock}.csv')

    # Convert the data into PyTorch tensors
    data_tensor = torch.tensor(data['Adj Close'].values, device=device)

    input_size = 100
    output_size = 1
    percentage_train = 80

    batch = data_split(data_tensor, input_size, output_size)

    # First shuffle before TRAIN and TEST splitting
    index_train = int((percentage_train/100)*batch.shape[0])
    train = batch[:index_train]
    test = batch[index_train:]

    # Set the hyperparameters
    hidden_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Create the RNN model
    model = RNN(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    max_data = torch.max(data_tensor)

    # Training loop
    for epoch in range(num_epochs):

        # Forward pass
        outputs = model(train[:,:input_size]/max_data)
        loss = criterion(outputs, train[:,input_size:]/max_data)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Generate predictions
    model.eval()
    with torch.no_grad():
        inputs = test[:,:input_size]
        predictions = model(inputs/max_data)*max_data#.squeeze().numpy()

    test = test.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    batch = batch.detach().cpu().numpy()
    data_tensor = data_tensor.detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot( range(data_tensor.shape[0]) , data_tensor, label='real')
    # plt.plot( range(input_size,input_size+output_size), predictions[i], color='orange', label='forecast')
    plt.plot( range(index_train,batch.shape[0]), predictions, color='orange', label='forecast')
    plt.axvline(x=input_size, color='red', linestyle='--')
    plt.title(stock)
    plt.legend()
    plt.show()

    print("Predictions:", predictions)



























