import datetime
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
train_df = pd.read_csv("../nlp-getting-started/train.csv")
test_df = pd.read_csv("../nlp-getting-started/test.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].values, train_df["target"].values, test_size=0.3)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.3)


class TwitterDisasterDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['target']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }


# Initialize the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', num_labels=2)

# Create the dataset instances
max_length = 128
train_dataset = TwitterDisasterDataset(train_df, tokenizer, max_length)
val_dataset = TwitterDisasterDataset(val_df, tokenizer, max_length)

# Create the data loaders
batch_size = 16
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 4
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        model.zero_grad()
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

    # Calculate the average training loss
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_val_loss += loss.item()

            # Calculate the number of correct predictions
            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == labels).item()

    # Calculate the average validation loss and accuracy
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = total_correct / len(val_dataset)

    print(f'\nEpoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}\n')

model_path = "../nlp-getting-started/roberta_twitter_disaster"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


def predict(model, test_data, tokenizer, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for text in test_data:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            preds.append(pred)

    return preds


# Load the test dataset
test_data_path = '../nlp-getting-started/test.csv'
test_df = pd.read_csv(test_data_path)
test_data = test_df["text"].values

# Get predictions on the test dataset
test_preds = predict(model, test_data, tokenizer, device)

# Create the submission DataFrame
submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})

# Save the submission DataFrame as a CSV file
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
filename = "../nlp-getting-started/submission_" + timestamp + ".csv"
submission_df.to_csv(filename, index=False)
