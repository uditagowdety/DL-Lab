import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

class RegressionModel(nn.Module):
    def __init__(self,nfeat):
        super(RegressionModel,self).__init__()
        self.linear=nn.Linear(nfeat,1)
    
    def forward(self,x):
        return self.linear(x)
    
class RegressionDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y.view(-1,1)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    
x = torch.tensor([
    [5.0, 1.0], 
    [7.0, 2.0], 
    [12.0, 3.0], 
    [16.0, 4.0], 
    [20.0, 5.0]
])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
dataset=RegressionDataset(x,y)
dataloader=DataLoader(dataset, batch_size=2,shuffle=True)

nfeat=x.shape[1]
model=RegressionModel(nfeat)
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.001)

epochs=100
loss_list=[]

for epoch in range(epochs):
    epoch_loss=0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        yp=model(batch_x)
        loss=criterion(yp,batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    
    loss_list.append(epoch_loss/len(dataloader))

    if epoch%10==0:
        print(f"loss={epoch_loss/len(dataloader)}")

plt.plot(loss_list, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.show()
