# train_phonetic_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import nltk
import json
from g2p_en import G2p
from sklearn.model_selection import train_test_split

nltk.download('cmudict')

# --- Phoneme Conversion ---
g2p = G2p()

def phrase_to_phonemes(phrase):
    return [ph for ph in g2p(phrase) if ph not in [" ", ".", ","]]

# Load and prepare dataset
df = pd.read_csv("brand_pairs.csv")
all_names = set(df["name1"]).union(df["name2"])

# Build vocab
phonemes_set = set()
for name in all_names:
    phonemes_set.update(phrase_to_phonemes(name))

phoneme2idx = {ph: i + 1 for i, ph in enumerate(sorted(phonemes_set))}
phoneme2idx["UNK"] = 0

def encode(phoneme_list, maxlen=20):
    idxs = [phoneme2idx.get(ph, 0) for ph in phoneme_list]
    return idxs[:maxlen] + [0] * (maxlen - len(idxs))

# Dataset
class BrandDataset(Dataset):
    def __init__(self, df, phoneme_map):
        self.df = df
        self.phoneme_map = phoneme_map
        self.all_names = list(phoneme_map.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name1 = row['name1']
        name2 = row['name2']
        label = 1

        if random.random() < 0.5:
            name2 = random.choice(self.all_names)
            while name2 in (row['name1'], row['name2']):
                name2 = random.choice(self.all_names)
            label = 0

        x1 = encode(self.phoneme_map[name1])
        x2 = encode(self.phoneme_map[name2])
        return torch.tensor(x1), torch.tensor(x2), torch.tensor(label, dtype=torch.float32)

# Model
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

class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, x1, x2):
        return self.sim(self.encoder(x1), self.encoder(x2))

# Train
def train(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataloader:
            optimizer.zero_grad()
            out = model(x1, x2)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"✅ Epoch {epoch+1}: Loss = {total_loss:.4f}")

if __name__ == "__main__":
    print("✅ Preprocessing phonemes...")
    phoneme_map = {name: phrase_to_phonemes(name) for name in all_names}

    train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
    train_ds = BrandDataset(train_df, phoneme_map)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    encoder = PhonemeEncoder(len(phoneme2idx))
    model = SiameseNetwork(encoder)

    print("🚀 Starting training...")
    train(model, train_dl, epochs=10)

    torch.save(model.state_dict(), "phonetic_model.pt")
    with open("phoneme2idx.json", "w") as f:
        json.dump(phoneme2idx, f)
    print("✅ Saved: phonetic_model.pt and phoneme2idx.json")
