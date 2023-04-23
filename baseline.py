import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
import torch
import numpy as np
from torch import nn


class RNN(torch.nn.Module):
    def __init__(self, hidden_units=64, dropout_rate=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(GLOVE_DIM, hidden_units, 1, batch_first=True)
        self.linear = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x shape: [batch, max_word_length, embedding_length]
        emb = self.drop(x)
        output, _ = self.rnn(emb)
        output = output[:, -1]
        output = self.linear(output)
        output = self.sigmoid(output)

        return output


# Load GloVe embeddings
GLOVE_DIM = 100
glove = GloVe(name="6B", dim=GLOVE_DIM)

# Preprocess data


def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    return tokens


def text_to_indices(text, max_length=64):
    tokens = preprocess_text(text)
    indices = [glove.stoi[t] for t in tokens if t in glove.stoi]
    indices += [0] * (max_length - len(indices))
    return indices[:max_length]


def create_dataset(data, targets, max_length=64):
    dataset = []
    for text, target in zip(data, targets):
        indices = text_to_indices(text, max_length)
        dataset.append(
            (torch.tensor(indices), torch.tensor(target, dtype=torch.float32)))
    return dataset


train_df = pd.read_csv("nlp-getting-started/train.csv")
test_df = pd.read_csv("nlp-getting-started/test.csv")

X = train_df["text"].values
y = train_df["target"].values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

train_dataset = create_dataset(X_train, y_train)
val_dataset = create_dataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define training and evaluation functions


def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(glove.vectors[inputs])
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(glove.vectors[inputs])
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            preds = (outputs.squeeze() > 0.5).float()
            correct += (preds == targets).float().sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)


# Train model and evaluate its performance
n_epochs = 10
for epoch in range(n_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

# Make predictions on test data


def predict(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in data:
            inputs = torch.tensor(text_to_indices(
                text)).unsqueeze(0).to(device)
            output = model(glove.vectors[inputs])
            pred = int(output.squeeze().item() > 0.5)
            predictions.append(pred)
    return predictions


test_data = test_df["text"].values
test_preds = predict(model, test_data, device)

# Create submission file
submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
submission_df.to_csv("nlp-getting-started/submission.csv", index=False)
