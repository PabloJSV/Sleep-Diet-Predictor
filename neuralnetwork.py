# Importing libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the sleep data
sleep_Data = pd.read_csv('AutoSleep-data.csv')

# Creating a new column called 'Date' getting only the date from ISO8601 column
sleep_Data['Date'] = sleep_Data['ISO8601'].str.split('T').str[0]

# Creating a new dataframe with only the Date and Dormido columns
sleep_Data = sleep_Data[['Date', 'dormido']]

# Remove rows with non-numeric values in the 'dormido' column
sleep_Data = sleep_Data[pd.to_numeric(sleep_Data['dormido'], errors='coerce').notnull()]

# Converting the 'Date' column to a Unix timestamp
sleep_Data['Date'] = pd.to_datetime(sleep_Data['Date']).astype(np.int64) // 10**9

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sleep_Data['Date'], sleep_Data['dormido'], test_size=0.2, random_state=42)

# Creating a neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = Net()

# Defining the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(zip(X_train, y_train)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().reshape(-1, 1)
        labels = torch.tensor(float(labels)).float().reshape(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch: %d, Loss: %.3f' % (epoch + 1, running_loss / len(X_train)))

# Evaluating the model
with torch.no_grad():
    y_pred = model(torch.tensor(X_test.values).float().reshape(-1, 1))
    y_pred = np.round(y_pred.numpy())
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
