# NOTES
# -----------
# **Step 1: Load MNIST Dataset**
# **Step 2: Define the Feedforward Neural Network**
# **Step 3: Initialize Model, Loss, and Optimizer**
# **Step 4: Training Loop**
# **Step 5: Evaluating the Model**
# **Step 6: Display Confusion Matrix**
# **Step 7: Check Learnable Parameters**
# -----------

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

loss_list=[]
torch.manual_seed(42)

x=torch.tensor([[0,0],
                [0,1],
                [1,0],
                [1,1]], dtype=torch.float32
                )
y=torch.tensor([0,1,1,0],dtype=torch.float32)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel,self).__init__()
        # self.linear1=nn.Linear(2,2,bias=True)
        # self.activation1=nn.Sigmoid()
        # self.linear2=nn.Linear(2,1,bias=True)
        self.model=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.Softmax() #not needed since we're using cross-entropy loss
        )
        
    def forward(self,x):
        # x=self.linear1(x)
        # x=self.activation1(x)
        # x=self.linear2(x)
        # return x

        return self.model(x)
    
class MyDataset(Dataset):
    def __init__(self,x,y):
        self.X=x
        self.Y=y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index].to(device), self.Y[index].to(device)


full_dataset=MyDataset(x,y)
batch_size=1

train_data_loader=DataLoader(full_dataset,batch_size=batch_size,shuffle=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=XORModel().to(device)
print(model)

loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.03)

def train_one_epoch(epoch):
    total_loss=0
    for i, data in enumerate(train_data_loader):
        inputs, labels=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_fn(outputs.flatten(),labels)
        loss.backward()

        optimizer.step()
        total_loss+=loss.item()
    
    return total_loss/len(train_data_loader)*batch_size


epochs=100

for epoch in range(epochs):
    model.train(True)
    avg_loss=train_one_epoch(epoch)
    loss_list.append(avg_loss)

    print(f"epoch {epoch+1}/{epochs}, loss = {avg_loss}")


print("\nmodel parameters:\n")
for param in model.named_parameters():
    print(param)

print()
input=torch.tensor([0,1],dtype=torch.float32)
model.eval()
output=model(input)

print(f"input: {input}, output: {output}")

plt.plot(loss_list)
plt.show()