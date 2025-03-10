import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

url="https://datahub.io/core/natural-gas/r/daily.csv"
df=pd.read_csv(url,parse_dates=["Date"])

df=df.sort_values("Date")
prices=df["Price"].values.reshape(-1,1)

scaler=MinMaxScaler()
prices=scaler.fit_transform(prices)

prices=np.array(prices,dtype=np.float32)

seq_length=10

def create_seq(data,seq_length):
    X,y=[],[]
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X),np.array(y)

X,y=create_seq(prices,seq_length)

train_size=int(len(X)*0.8)
X_train,X_test=X[:train_size],X[train_size:]
y_train,y_test=y[:train_size],y[train_size:]

X_train=torch.tensor(X_train).view(-1,seq_length,1)
y_train=torch.tensor(y_train).view(-1,1)
X_test=torch.tensor(X_test).view(-1,seq_length,1)
y_test=torch.tensor(y_test).view(-1,1)

print(f"train data shape: {X_train.shape}, {y_train.shape}")
print(f"test data shape: {X_test.shape}, {y_test.shape}")

class priceRNN(nn.Module):
    def __init__(self,input_size=1,hidden_size=10,num_layers=2,output_size=1):
        super(priceRNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.rnn=nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        out,_=self.rnn(x,h0)
        out=self.fc(out[:,-1,:])
        return out

model=priceRNN()
criterion=nn.MSELoss()
optimiser=optim.Adam(model.parameters(),lr=0.03)
epochs=10
loss_list=[]

for epoch in range(epochs):
    model.train()
    optimiser.zero_grad()
    y_pred=model(X_train)
    loss=criterion(y_pred,y_train)
    loss.backward()
    optimiser.step()

    loss_list.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_pred = scaler.inverse_transform(y_pred.numpy())
y_test = scaler.inverse_transform(y_test.numpy())

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="dashed")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.title("Natural Gas Price Prediction using RNN")
plt.show()