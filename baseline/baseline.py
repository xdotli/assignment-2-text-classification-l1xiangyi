# importing packages
import datetime
import numpy as np
import pandas as pd
import torch
import re
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe, build_vocab_from_iterator
from torchtext.data import get_tokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

# loading datasets and preprocessing


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_special_characters(text):
    special_chars_pattern = re.compile(r'[^a-zA-Z\s]')
    return special_chars_pattern.sub(r'', text)


def preprocess(df):
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(remove_urls)
    df['text'] = df['text'].apply(remove_special_characters)


train_file_path = "../nlp-getting-started/train.csv"
test_file_path = "../nlp-getting-started/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

preprocess(train_df)
preprocess(test_df)

# check if the data is successfully cleaned
test_df[test_df['id'] == 150]

# Create the datasets

tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
glove = torchtext.vocab.GloVe(name="6B", dim=100)


def build_vocab(tokenized_data, min_freq=5, specials=('<unk>', '<pad>')):
    def iterator():
        for tokens in tokenized_data:
            yield tokens

    vocab = build_vocab_from_iterator(
        iterator(), min_freq=min_freq, specials=specials, special_first=True)
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def load_and_tokenize(file_path, tokenizer):
    df = pd.read_csv(file_path)
    text_data = [(tokenizer(row["text"]), row["target"])
                 for _, row in df.iterrows()]
    return text_data


def create_dataset(file_path, tokenizer, vocab=None):
    text_data = load_and_tokenize(file_path, tokenizer)

    if vocab is None:
        vocab = build_vocab([tokens for tokens, _ in text_data])

    data = []
    for tokens, label in text_data:
        data.append((torch.tensor([vocab[token]
                    for token in tokens]), torch.tensor(label)))

    return CustomDataset(data, vocab)


class CustomDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_df, val_df = train_test_split(train_df, test_size=0.3)
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

train_dataset = create_dataset("train_split.csv", tokenizer)
val_dataset = create_dataset("val_split.csv", tokenizer, train_dataset.vocab)


# Define the RNN and begin training

class RNN(torch.nn.Module):
    def __init__(self, hidden_units=64, dropout_rate=0.4):
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


# Instantiate the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = RNN()
model = model.to(device)

# Define the RNN and begin training

GLOVE_DIM = 100


class RNN(torch.nn.Module):
    def __init__(self, hidden_units=64, dropout_rate=0.4):
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


# Instantiate the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = RNN()
model = model.to(device)

embedding = nn.Embedding(len(train_dataset.vocab), GLOVE_DIM).to(device)

# Set up the loss function, optimizer, and learning rate scheduler
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Set up the DataLoader for batching
batch_size = 64


def pad_collate_fn(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)

    return xx_pad, torch.tensor(yy)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=pad_collate_fn)

# Train the model

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = embedding(x).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        y = y.float().view(-1, 1)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_loss = 0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = embedding(x).to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred = (y_pred > 0.5).float()
            val_correct += (y_pred == y.float().view(-1, 1)).sum().item()
            loss = loss_fn(y_pred, y.float().view(-1, 1))
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}: train_loss={total_loss / len(train_loader):.4f}, val_loss={val_loss / len(val_loader):.4f}, val_acc={val_correct / len(val_dataset):.4f}")

model_path = "../nlp-getting-started/baseline.pth"

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'embedding_state_dict': embedding.state_dict(),
    'vocab': train_dataset.vocab,
    'GLOVE_DIM': GLOVE_DIM,
}, model_path)

test_df = pd.read_csv('../nlp-getting-started/test.csv')


def predict(model, test_data, device):
    test_preds = []
    model.eval()
    with torch.no_grad():
        for x in test_data:
            x = torch.tensor([train_dataset.vocab[token] if token in train_dataset.vocab else train_dataset.vocab['<unk>']
                             for token in tokenizer(x)]).unsqueeze(0)
            x = embedding(x).to(device)
            y_pred = model(x)
            y_pred = (y_pred > 0.5).float()
            test_preds.append(int(y_pred.item()))
    return test_preds


preprocess(test_df)

test_data = test_df["text"].values
test_preds = predict(model, test_data, device)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_fn)

# test_preds = predict(model, test_loader, device)
submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
filename = "../nlp-getting-started/submission_" + timestamp + ".csv"
submission_df.to_csv(filename, index=False)
