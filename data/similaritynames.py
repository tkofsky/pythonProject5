import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')

cmu_dict = cmudict.dict()


# --- Utilities ---
def word_to_phonemes(word):
    word = word.lower()
    return cmu_dict.get(word, [["UNK"]])[0]  # fallback: [["UNK"]]


def phrase_to_phonemes(phrase):
    return [ph for word in phrase.split() for ph in word_to_phonemes(word)]


# Tokenizer and vocab
all_phonemes = set(ph for phs in cmu_dict.values() for variant in phs for ph in variant)
phoneme2idx = {ph: i + 1 for i, ph in enumerate(sorted(all_phonemes))}
phoneme2idx["UNK"] = 0


def encode(phoneme_list, maxlen=20):
    idxs = [phoneme2idx.get(ph, 0) for ph in phoneme_list]
    idxs = idxs[:maxlen] + [0] * (maxlen - len(idxs))
    return idxs


# --- Dataset ---
class BrandDataset(Dataset):
    def __init__(self, df, negative_sample=True):
        self.df = df
        self.negative_sample = negative_sample
        self.all_names = list(set(df['name1']).union(set(df['name2'])))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name1 = row['name1']
        name2 = row['name2']
        label = 1

        # Negative sampling
        if self.negative_sample and random.random() < 0.5:
            name2 = random.choice(self.all_names)
            while name2 in (row['name1'], row['name2']):
                name2 = random.choice(self.all_names)
            label = 0

        x1 = encode(phrase_to_phonemes(name1))
        x2 = encode(phrase_to_phonemes(name2))
        return torch.tensor(x1), torch.tensor(x2), torch.tensor(label, dtype=torch.float32)


# --- Model ---
class PhonemeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
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
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return self.sim(e1, e2)


# --- Training Loop ---
def train(model, dataloader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, label in dataloader:
            optimizer.zero_grad()
            score = model(x1, x2)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# --- Load & Run ---
# Load your CSV of matched names
df = pd.read_csv("brand_pairs.csv")  # make sure it has columns "name1", "name2"
train_df, val_df = train_test_split(df, test_size=0.1)

train_ds = BrandDataset(train_df)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

encoder = PhonemeEncoder(vocab_size=len(phoneme2idx))
model = SiameseNetwork(encoder)

train(model, train_dl, epochs=50)

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("brand_pairs.csv")  # ensure this file exists

    # Split into train/validation (you can later add val loop)
    train_df, _ = train_test_split(df, test_size=0.1, random_state=42)

    # Create Dataset and DataLoader
    train_ds = BrandDataset(train_df)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Create model
    encoder = PhonemeEncoder(vocab_size=len(phoneme2idx))
    model = SiameseNetwork(encoder)

    # Train
    train(model, train_dl, epochs=50)

    # Save the trained model
    torch.save(model.state_dict(), "phonetic_model.pt")

