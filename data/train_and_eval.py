import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from g2p_en import G2p
from sklearn.metrics import roc_auc_score, accuracy_score
import random

g2p = G2p()
MAXLEN = 20

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

def train_model(pairs, phoneme2idx, epochs=5):
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

def run_experiment():
    all_data = pd.read_csv("labeled_training_pairs.csv")
    test_df = pd.read_csv("test_pairs.csv")
    sizes = [10, 20, 50, 100, 200,300,400,500,600,1000,5000]

    print("Training Rows\tAUC\tAccuracy")
    for size in sizes:
        sample_df = all_data.sample(n=min(size, len(all_data)), random_state=42)
        sample_pairs = list(zip(sample_df["name1"], sample_df["name2"], sample_df["label"]))
        phoneme2idx = build_vocab(sample_pairs)
        model = train_model(sample_pairs, phoneme2idx)
        auc, acc = evaluate_model(model, phoneme2idx, test_df)
        print(f"{size}\t\t{auc:.2f}\t{acc:.2f}")

if __name__ == "__main__":
    run_experiment()
