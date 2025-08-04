import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from g2p_en import G2p
from sklearn.metrics import roc_auc_score, accuracy_score
import random
import numpy as np

g2p = G2p()
MAXLEN = 20
EPOCHS = 5
REPEATS = 3  # Number of times to repeat each training size for averaging

def phrase_to_phonemes(phrase):
    return [ph for ph in g2p(phrase) if ph not in [" ", ".", ","]]

def build_vocab(pairs):
    phonemes = set()
    for n1, n2, _ in pairs:
        phonemes.update(phrase_to_phonemes(n1))
        phonemes.update(phrase_to_phonemes(n2))
    phoneme2idx = {ph: i + 1 for i, ph in enumerate(sorted(phonemes))}
    phoneme2idx["PAD"] = 0
    return phoneme2idx

def encode(phonemes, phoneme2idx):
    idxs = [phoneme2idx.get(ph, 0) for ph in phonemes]
    return idxs[:MAXLEN] + [0] * (MAXLEN - len(idxs))

class LabeledBrandPairDataset(Dataset):
    def __init__(self, pairs, phoneme2idx):
        self.pairs = pairs
        self.phoneme2idx = phoneme2idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name1, name2, label = self.pairs[idx]
        ph1 = phrase_to_phonemes(name1)
        ph2 = phrase_to_phonemes(name2)
        enc1 = torch.tensor(encode(ph1, self.phoneme2idx))
        enc2 = torch.tensor(encode(ph2, self.phoneme2idx))
        return enc1, enc2, torch.tensor(label, dtype=torch.float)

class PhonemeEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = self.pool(out.transpose(1, 2)).squeeze(2)
        return pooled

class SiameseClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.sim = nn.CosineSimilarity(dim=1)
        self.out = nn.Sigmoid()

    def forward(self, x1, x2):
        sim_score = self.sim(self.encoder(x1), self.encoder(x2))
        return self.out(sim_score)

def train_model(pairs, phoneme2idx, epochs=EPOCHS):
    dataset = LabeledBrandPairDataset(pairs, phoneme2idx)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SiameseClassifier(PhonemeEncoder(len(phoneme2idx)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        for x1, x2, y in loader:
            pred = model(x1, x2)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, phoneme2idx, test_df):
    y_true, y_scores = [], []
    model.eval()
    for _, row in test_df.iterrows():
        ph1 = phrase_to_phonemes(row["name1"])
        ph2 = phrase_to_phonemes(row["name2"])
        enc1 = torch.tensor([encode(ph1, phoneme2idx)])
        enc2 = torch.tensor([encode(ph2, phoneme2idx)])
        with torch.no_grad():
            score = model(enc1, enc2).item()
        y_scores.append(score)
        y_true.append(row["label"])
    auc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, [1 if s >= 0.5 else 0 for s in y_scores])
    return auc, acc

def generate_balanced_data(df, n):
    positives = list(zip(df[df["label"] == 1]["name1"], df[df["label"] == 1]["name2"]))
    positives = random.sample(positives, min(n // 2, len(positives)))
    all_names = list(set(df["name1"]).union(set(df["name2"])))
    negatives = set()
    while len(negatives) < len(positives):
        a, b = random.sample(all_names, 2)
        if (a, b) not in positives and (b, a) not in positives:
            negatives.add((a, b))
    combined = [(a, b, 1) for a, b in positives] + [(a, b, 0) for a, b in negatives]
    random.shuffle(combined)
    return combined

def run_experiment():
    full_df = pd.read_csv("labeled_training_pairs.csv")
    test_df = pd.read_csv("test_pairs.csv")
    sizes = [10, 20, 50, 100, 200, 300, 500, 1000]
    print("Training Rows\tAUC (avg)\tAccuracy (avg)")
    for size in sizes:
        aucs, accs = [], []
        for _ in range(REPEATS):
            pairs = generate_balanced_data(full_df, size)
            phoneme2idx = build_vocab(pairs)
            model = train_model(pairs, phoneme2idx)
            auc, acc = evaluate_model(model, phoneme2idx, test_df)
            aucs.append(auc)
            accs.append(acc)
        print(f"{size}\t\t{np.mean(aucs):.2f}\t\t{np.mean(accs):.2f}")

run_experiment()

