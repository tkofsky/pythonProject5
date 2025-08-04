# predict.py

import torch
import torch.nn as nn
from g2p_en import G2p
import json
import nltk

nltk.download('cmudict')
g2p = G2p()

# Load vocab
with open("phoneme2idx.json", "r") as f:
    phoneme2idx = json.load(f)

def phrase_to_phonemes(phrase):
    return [ph for ph in g2p(phrase) if ph not in [" ", ".", ","]]

def encode(phoneme_list, maxlen=20):
    idxs = [phoneme2idx.get(ph, 0) for ph in phoneme_list]
    return idxs[:maxlen] + [0] * (maxlen - len(idxs))

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

# Load model
encoder = PhonemeEncoder(len(phoneme2idx))
model = SiameseNetwork(encoder)
model.load_state_dict(torch.load("phonetic_model.pt"))
#model.eval()

# Predict
def predict_similarity(name1, name2):
    ph1 = phrase_to_phonemes(name1)
    ph2 = phrase_to_phonemes(name2)

    print(f"\n{name1} → {ph1}")
    print(f"{name2} → {ph2}")

    x1 = torch.tensor([encode(ph1)])
    x2 = torch.tensor([encode(ph2)])
    print("Encoded 1:", x1.tolist()[0])
    print("Encoded 2:", x2.tolist()[0])

    with torch.no_grad():
        score = model(x1, x2).item()
        normalized = (score + 1) / 2
        print(f"Similarity Score: {normalized:.4f}")
        return normalized

if __name__ == "__main__":
    pairs = [
        ("NyteStryke", "Night Strike"),
        ("ByteRide", "Bright Ride"),
        ("LiteSpeed", "Light Speed"),
        ("Nike", "SpeedMax"),
        ("GlowForce", "Go Force")
    ]

    for a, b in pairs:
        predict_similarity(a, b)
