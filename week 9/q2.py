import os
import glob
import string
import unicodedata
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------
# 1. Setup
# -------------------------
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in all_letters)

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip()) for line in f]

category_lines = {}
all_categories = []

for filename in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    category_lines[category] = read_lines(filename)

n_categories = len(all_categories)
if n_categories == 0:
    raise RuntimeError("No data found in 'data/names/' folder.")

# -------------------------
# 2. Encoding
# -------------------------
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][all_letters.find(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# -------------------------
# 3. LSTM Model
# -------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.fc(lstm_out[-1])
        return self.softmax(output)

# -------------------------
# 4. Training
# -------------------------
criterion = nn.NLLLoss()
learning_rate = 0.005
lstm_model = LSTMClassifier(n_letters, 128, n_categories)

def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def random_training_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    lstm_model.zero_grad()
    output = lstm_model(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in lstm_model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

# -------------------------
# 5. Training Loop
# -------------------------
n_iters = 20000
print_every = 1000
current_loss = 0
all_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f'{iter} {loss:.4f} {line} → {guess} {correct}')
        all_losses.append(current_loss / print_every)
        current_loss = 0

# -------------------------
# 6. Prediction
# -------------------------
def predict(input_line, n_predictions=3):
    print(f'\n> {input_line}')
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        output = lstm_model(line_tensor)

        topv, topi = output.topk(n_predictions, 1)
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print(f'{all_categories[category_index]} ({value:.2f})')

# Test predictions
predict("Sundar")
predict("Sakamoto")
predict("Schmidt")
