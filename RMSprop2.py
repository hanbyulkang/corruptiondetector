#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import re
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if len(sys.argv) > 4:
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    embedding_dim = int(sys.argv[3])
    hidden_dim = int(sys.argv[4])
else:
    print("Insufficient arguments provided. Using default values.")
    batch_size = 64  # 기본 배치 크기
    num_epochs = 10  # 기본 에포크 수
    embedding_dim = 128  # 기본 임베딩 차원
    hidden_dim = 256  # 기본 히든 차원

print(f"Batch size: {batch_size}, Num epochs: {num_epochs}, Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")

# 데이터 전처리 및 로딩 클래스
class TextPairDataset(Dataset):
    def __init__(self, filepath, vocab=None, is_train=True):
        self.vocab = vocab if vocab is not None else self.build_vocab(filepath, is_train)
        self.pairs = self.load_dataset(filepath, is_train)
        
    def load_dataset(self, filepath, is_train):
        pairs = []
        with open(filepath, 'r', encoding='latin1') as file:
            for line in file:
                parts = line.strip().split('\t')
                if is_train:
                    original, corrupted = parts
                    pairs.append((original, corrupted, 0))  # 0 for original
                    pairs.append((corrupted, original, 1))  # 1 for corrupted
                else:
                    pairs.append((parts[0], parts[1]))
        return pairs
    
    def build_vocab(self, filepath, is_train):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        with open(filepath, 'r', encoding='latin1') as file:
            for line in file:
                parts = line.strip().split('\t')
                sentences = parts if not is_train else [parts[0]]
                for sentence in sentences:
                    for word in sentence.split():
                        if word not in vocab:
                            vocab[word] = len(vocab)
        return vocab
    
    def sentence_to_tensor(self, sentence):
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence.split()]
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if len(self.pairs[idx]) == 3:
            original, corrupted, label = self.pairs[idx]
            return self.sentence_to_tensor(original), self.sentence_to_tensor(corrupted), label
        else:
            return self.sentence_to_tensor(self.pairs[idx][0]), self.sentence_to_tensor(self.pairs[idx][1])

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# collate_fn 함수
def collate_fn(batch):
    originals, corrupteds, labels = zip(*batch)
    originals_padded = pad_sequence([torch.tensor(o) for o in originals], batch_first=True, padding_value=0)
    corrupteds_padded = pad_sequence([torch.tensor(c) for c in corrupteds], batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return originals_padded, corrupteds_padded, labels

# 모델 정의
class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        
    def forward(self, original, corrupted):
        original_embedded = self.embedding(original)
        corrupted_embedded = self.embedding(corrupted)
        _, (original_hidden, _) = self.lstm(original_embedded)
        _, (corrupted_hidden, _) = self.lstm(corrupted_embedded)
        combined_hidden = torch.cat((original_hidden[-1], corrupted_hidden[-1]), dim=1)
        output = self.fc(combined_hidden)
        return output
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, early_stopping_patience=3):
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path='checkpoint.pt')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (original, corrupted, labels) in enumerate(train_loader, 1):
            original, corrupted, labels = original.to(device), corrupted.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(original, corrupted)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                sys.stdout.write(f'\rEpoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()}')
                sys.stdout.flush()

        # 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for original, corrupted, labels in val_loader:
                original, corrupted, labels = original.to(device), corrupted.to(device), labels.to(device)
                outputs = model(original, corrupted)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'\nEpoch: {epoch+1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}')

        # Early Stopping 검사
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 가장 좋은 모델 로드
    model.load_state_dict(torch.load('checkpoint.pt'))

    
def calculate_accuracy(model, data_loader):
    model.eval()  # 모델을 평가 모드로 설정
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # 기울기 계산을 비활성화
        for original, corrupted, labels in data_loader:
            original, corrupted = original.to(device), corrupted.to(device)
            labels = labels.to(device)
            outputs = model(original, corrupted)
            _, predicted = torch.max(outputs, 1)  # 확률이 가장 높은 클래스를 예측값으로 선택
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

    accuracy = 100 * correct_preds / total_preds
    return accuracy


print("preapring dataset...")
full_dataset = TextPairDataset('train.txt', is_train=True)
vocab = full_dataset.vocab

train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print("preaparing data loader...")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

print("initializing model ... ")
model = Classifier(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

print("start training ... ")
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, early_stopping_patience=3)

print("save files ...")
with torch.no_grad(), open('part1.txt', 'w') as f:
    model.eval()
    for original, corrupted in test_loader:
        original, corrupted = original.to(device), corrupted.to(device)
        outputs = model(original, corrupted)
        predictions = outputs.argmax(dim=1)
        for pred in predictions.cpu().numpy():
            f.write(f'{"A" if pred == 0 else "B"}\n')


# In[ ]:




