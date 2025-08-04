# predict.py

import torch
import torch.nn as nn
import nltk
from nltk.corpus import cmudict
from g2p_en import G2p
g2p = G2p()

nltk.download('cmudict')
cmu_dict = cmudict.dict()

# --- Phoneme Vocab ---
all_phonemes = set(ph for phs in cmu_dict.values() for variant in phs for ph in variant)
phoneme2idx = {ph: i+1 for i, ph in enumerate(sorted(all_phonemes))}
phoneme2idx["UNK"] = 0

# --- Utilities ---
def word_to_phonemes(word):
    word = word.lower()
    return cmu_dict.get(word, [["UNK"]])[0]

#def phrase_to_phonemes(phrase):
#    return [ph for word in phrase.split() for ph in word_to_phonemes(word)]
def phrase_to_phonemes(phrase):
    phonemes = g2p(phrase)
    return [ph for ph in phonemes if ph not in [" ", ".", ","]]




def encode(phoneme_list, maxlen=20):
    idxs = [phoneme2idx.get(ph, 0) for ph in phoneme_list]
    return idxs[:maxlen] + [0] * (maxlen - len(idxs))

# --- Model ---
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

# --- Load Model ---
encoder = PhonemeEncoder(len(phoneme2idx))
model = SiameseNetwork(encoder)
model.load_state_dict(torch.load("phonetic_model.pt"))
model.eval()

# --- Predict + Debug ---
def predict_similarity(name1, name2):
    ph1 = phrase_to_phonemes(name1)
    ph2 = phrase_to_phonemes(name2)

    print(f"\n{name1} → Phonemes:", ph1)
    print(f"{name2} → Phonemes:", ph2)

    x1 = torch.tensor([encode(ph1)])
    x2 = torch.tensor([encode(ph2)])

    print("Encoded 1:", x1.tolist()[0])
    print("Encoded 2:", x2.tolist()[0])

    with torch.no_grad():
        score = model(x1, x2).item()
        normalized = (score + 1) / 2  # convert cosine [-1,1] to [0,1]
        print(f"Raw cosine score: {score:.4f}")
        print(f"Normalized similarity score: {normalized:.4f}")
        return normalized

# --- Try Out Some Pairs ---
if __name__ == "__main__":
    pairs = [
        ("NyteStryke", "Night Strike"),
        ("ByteRide", "Bright Ride"),
        ("LiteSpeed", "Light Speed"),
        ("Nike", "SpeedMax"),
        ("GlowForce", "Go Force"),
    ]
    for a, b in pairs:
        score = predict_similarity(a, b)
