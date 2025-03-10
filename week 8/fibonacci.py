import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Fibonacci Data
def generate_fibonacci(n):
    fib_sequence = [0, 1]
    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

train_data = np.array(generate_fibonacci(100), dtype=np.float64)
test_data = np.array(generate_fibonacci(120)[100:], dtype=np.float64)

max_fib = train_data[-1]  # Normalize using the last training value
train_data = torch.tensor(train_data / max_fib, dtype=torch.float32).view(-1, 1)
test_data = torch.tensor(test_data / max_fib, dtype=torch.float32).view(-1, 1)

# 2. Define RNN Model
class FibRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(FibRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, hidden = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 3. Train RNN Model
input_size = 1
hidden_size = 10
output_size = 1
epochs = 100
learning_rate = 0.1
loss_list = []

model = FibRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_input = train_data[:-1].view(-1, 1, 1)
train_target = train_data[1:].view(-1, 1)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_input)
    loss = criterion(output, train_target)
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.6f}")

# 4. Test RNN Model (Print Predictions Instead of Plotting)
model.eval()
preds = []
input_seq = test_data[0].view(1, 1, 1)

with torch.no_grad():
    for x in range(len(test_data) - 1):
        pred = model(input_seq)
        preds.append(pred.item())
        input_seq = pred.view(1, 1, 1)

# Convert predictions back to original Fibonacci scale
preds = np.array(preds) * max_fib
true_seq = test_data.numpy().flatten() * max_fib

plt.figure(figsize=(8, 5))
plt.plot(loss_list, label="training loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("training loss")
plt.legend()
plt.show()
