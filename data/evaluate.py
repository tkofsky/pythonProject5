# evaluate.py

import torch
import torch.nn as nn
import pandas as pd
import json
from g2p_en import G2p
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import sys

g2p = G2p()

# --- Phoneme utilities ---
def phrase_to_phonemes(phrase):
    return [ph for ph in g2p(phrase) if ph not in [" ", ".", ","]]

def encode(phoneme_list, phoneme2idx, maxlen=20):
    idxs = [phoneme2idx.get(ph, 0) for ph in phoneme_list]
    return idxs[:maxlen] + [0] * (maxlen - len(idxs))

# --- Model definitions ---
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

# --- Predict one pair ---
def predict_similarity(name1, name2, model, phoneme2idx):
    ph1 = phrase_to_phonemes(name1)
    ph2 = phrase_to_phonemes(name2)
    x1 = torch.tensor([encode(ph1, phoneme2idx)])
    x2 = torch.tensor([encode(ph2, phoneme2idx)])
    with torch.no_grad():
        score = model(x1, x2).item()
        return (score + 1) / 2  # Normalize cosine [-1,1] → [0,1]

# --- Evaluate entire model on test set ---
def evaluate_model(model_path, phoneme_path, test_path):
    with open(phoneme_path, "r") as f:
        phoneme2idx = json.load(f)

    encoder = PhonemeEncoder(len(phoneme2idx))
    model = SiameseNetwork(encoder)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    df = pd.read_csv(test_path)
    y_true = []
    y_scores = []

    for _, row in df.iterrows():
        score = predict_similarity(row["name1"], row["name2"], model, phoneme2idx)
        y_scores.append(score)
        y_true.append(row["label"])

    auc = roc_auc_score(y_true, y_scores)
    preds = [1 if s >= 0.5 else 0 for s in y_scores]
    acc = accuracy_score(y_true, preds)

    print(f"🔍 Model: {model_path}")
    print(f"✅ AUC: {auc:.4f}")
    print(f"✅ Accuracy: {acc:.4f}")
    return auc

# --- Batch Mode to Plot Multiple Sizes ---
def evaluate_multiple(sizes, test_path):
    results = []
    for size in sizes:
        model_path = f"phonetic_model_{size}.pt"
        phoneme_path = f"phoneme2idx_{size}.json"
        auc = evaluate_model(model_path, phoneme_path, test_path)
        results.append((size, auc))

    # Plot
    sizes, aucs = zip(*results)
    plt.plot(sizes, aucs, marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.title("Model Performance vs. Training Size")
    plt.grid(True)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    # Hardcode your test file
    test_file = "test_pairs.csv"

    # Option 1 – single model
    evaluate_model("phonetic_model.pt", "phoneme2idx.json", test_file)

    # Option 2 – run multiple experiments (uncomment if needed)
    # sizes = [50, 100, 250, 500]
    # evaluate_multiple(sizes, test_file)