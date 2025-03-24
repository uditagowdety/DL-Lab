import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# ---------------------------
# 1. Define character set and encoding
# ---------------------------
all_chars = string.ascii_lowercase + " .,;'"
n_chars = len(all_chars)

def char_to_tensor(char):
    tensor = torch.zeros(1, n_chars)
    tensor[0][all_chars.find(char)] = 1
    return tensor

def string_to_tensor(string):
    tensor = torch.zeros(len(string), 1, n_chars)
    for i, char in enumerate(string):
        tensor[i][0][all_chars.find(char)] = 1
    return tensor

# ---------------------------
# 2. Define the LSTM model
# ---------------------------
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden()
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output[-1])
        return output, hidden

    def init_hidden(self):
        # (num_layers, batch, hidden_size)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# ---------------------------
# 3. Training data preparation
# ---------------------------
data = "hello world hello lstm text prediction using characters"
seq_length = 5
examples = []
targets = []

for i in range(len(data) - seq_length):
    seq = data[i:i+seq_length]
    target = data[i+seq_length]
    if all(c in all_chars for c in seq + target):
        examples.append(seq)
        targets.append(target)

# ---------------------------
# 4. Model setup
# ---------------------------
hidden_size = 128
model = CharLSTM(n_chars, hidden_size, n_chars)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ---------------------------
# 5. Training loop
# ---------------------------
epochs = 200

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(examples)):
        seq_tensor = string_to_tensor(examples[i])
        target_index = all_chars.find(targets[i])
        target_tensor = torch.tensor([target_index])

        output, hidden = model(seq_tensor)

        loss = criterion(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(examples):.4f}")

# ---------------------------
# 6. Prediction function
# ---------------------------
def predict_next(input_seq, top_k=3):
    with torch.no_grad():
        input_tensor = string_to_tensor(input_seq)
        output, _ = model(input_tensor)
        probs = torch.softmax(output, dim=1)

        topv, topi = probs.topk(top_k)
        predictions = [(all_chars[topi[0][i]], topv[0][i].item()) for i in range(top_k)]
        return predictions

# Test it
test_seq = "hello"
predictions = predict_next(test_seq)
print(f"\nNext character predictions for '{test_seq}':")
for char, score in predictions:
    print(f"{char} ({score:.2f})")
