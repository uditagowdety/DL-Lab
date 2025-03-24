import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import MSELoss
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

df=pd.read_csv("./daily.csv")
df=df.dropna()

y=df['Price'].values
x=np.arange(1,len(y),1)

print("len of y: ",len(y))

minm=y.min()
maxm=y.max()
print(minm, maxm)
y=(y-minm)/(maxm-minm)

seq_len=10

X=[]
Y=[]

for i in range(0,5900):
    list1=[]
    for j in range(i, i+seq_len):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[j+1])

X=np.array(X)
Y=np.array(Y)

x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.10, random_state=42, shuffle=False, stratify=None)

class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len=x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len

dataset=NGTimeSeries(x_train,y_train)
train_loader=DataLoader(dataset, shuffle=True,batch_size=256)

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel,self).__init__()
        self.lstm=nn.LSTM(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
        self.fc1=nn.Linear(in_features=5, out_features=1)

    def forward(self,x):
        output, _status=self.lstm(x)
        output=output[:,-1,:]
        output=self.fc1(torch.relu(output))
        return output

model=LSTMModel()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=100

for i in range(epochs):
    for j, (xb, yb) in enumerate(train_loader):
        xb = xb.view(-1, seq_len, 1)
        y_pred = model(xb).reshape(-1)
        loss = criterion(y_pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%10==0:
        print(i, "th iteration: ",loss.item())


model.eval()
with torch.no_grad():
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, seq_len, 1)
    test_pred = model(x_test_tensor).view(-1)

plt.plot(test_pred.numpy(), label="predicted")
plt.plot(y_test, label="original")
plt.legend()
plt.show()

y = y * (maxm - minm) + minm
y_pred = test_pred.numpy() * (maxm - minm) + minm
plt.plot(y, label="Full Original Series")
plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label="Predicted")
plt.legend()
plt.show()