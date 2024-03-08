import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class for loading and processing data
class TextPairDataset(Dataset):
    def __init__(self, filepath, vocab=None, is_train=True):
        # Initialize vocabulary or build it if not provided
        self.vocab = vocab if vocab is not None else self.build_vocab(filepath)
        # Load dataset accordingly if it's for training or testing
        if is_train:
            self.sentences, self.labels = self.load_dataset(filepath, is_train)
        else:  # Handling test data
            self.sentences = self.load_dataset(filepath, is_train)
    
    def load_dataset(self, filepath, is_train):
        # Load training data with labels
        if is_train:
            sentences, labels = [], []
            with open(filepath, 'r', encoding='latin1') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    original, corrupted = parts[0], parts[1]
                    sentences.append(original)
                    labels.append(0)  # Label 0 for original sentences
                    sentences.append(corrupted)
                    labels.append(1)  # Label 1 for corrupted sentences
            return sentences, labels
        else:  # Load test data without labels
            sentences = []
            with open(filepath, 'r', encoding='latin1') as file:
                for line in file:
                    sentence = line.strip()  # Remove newline character
                    sentences.append(sentence)
            return sentences
    
    def build_vocab(self, filepath):
        # Build vocabulary from the dataset
        vocab = {'<PAD>': 0, '<UNK>': 1}
        with open(filepath, 'r', encoding='latin1') as file:
            for line in file:
                for word in line.strip().split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
        return vocab
    
    def sentence_to_tensor(self, sentence):
        # Convert sentence to tensor by converting each word to its index in the vocab
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence.split()]
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.sentences)
    
    def __getitem__(self, idx):
        # Get item by index
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return self.sentence_to_tensor(sentence), label

# Classifier model definition
class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, sentences):
        # Forward pass through the network
        embedded = self.embedding(sentences)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

# EarlyStopping class for managing early stopping
class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=0, path='model_checkpoint.pt'):
        # Initialize early stopping parameters
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0
    
    def __call__(self, val_loss, model):
        # Check if early stopping is triggered
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        # Save model checkpoint
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} to {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def collate_fn(batch):
    # Collate function to pad sentences
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return sentences_padded, labels

# Function to train the model with validation and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, early_stopping_patience=3):
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path='model_checkpoint.pt')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch_idx, (sentences, labels) in enumerate(train_loader, 1):
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                sys.stdout.write(f'\rEpoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}')
                sys.stdout.flush()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sentences, labels in val_loader:
                sentences, labels = sentences.to(device), labels.to(device)
                outputs = model(sentences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'\nEpoch: {epoch+1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}')

        # Early Stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model
    model.load_state_dict(torch.load('model_checkpoint.pt'))

# Main logic for model training and evaluation
def main():
    # Prepare dataset and dataloader
    dataset = TextPairDataset('train.txt', is_train=True)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    model = Classifier(len(dataset.vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())  # Using RMSprop optimizer

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

    # Assuming you have logic to prepare your test dataset
    test_dataset = TextPairDataset('test.rand.txt', vocab=dataset.vocab, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Save predictions for the test dataset
    model.eval()
    predictions = []
    with torch.no_grad():
        for sentences, _ in test_loader:
            sentences = sentences.to(device)
            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().tolist())

    # Save prediction results to part1.txt
    with open('part1.txt', 'w') as f:
        for pred in predictions:
            f.write(f'{"A" if pred == 0 else "B"}\n')

if __name__ == '__main__':
    main()
