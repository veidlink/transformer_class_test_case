import math
import torch
import torch.nn as nn
import razdel
from torch.nn import functional as F

class Config:
    n_head = 8 #является делителем размерности вложения (n_embd), поэтому меняю (32 делится на 8)
    n_embd = 32 #меняю чтобы соответствовало MAX_LEN ниже
    n_layer = 12
    seq_len = 32
    embd_pdrop = 0.5
    resid_pdrop = 0.5
    attn_pdrop = 0.5
    vocab_size = 1024

class Attention(nn.Module):
    def __init__(self):  # <-- исправляем здесь
        super(Attention, self).__init__()  # <-- исправляем здесь

        assert Config.n_embd % Config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(Config.n_embd, Config.n_embd)
        self.query = nn.Linear(Config.n_embd, Config.n_embd)
        self.value = nn.Linear(Config.n_embd, Config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(Config.attn_pdrop)
        self.resid_drop = nn.Dropout(Config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        
    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, Config.n_head, C // Config.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, Config.n_head, C // Config.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, Config.n_head, C // Config.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self):  # <-- исправляем здесь
        super(TransformerBlock, self).__init__()  # <-- исправляем здесь
        self.norm1 = nn.BatchNorm1d(Config.n_embd)
        self.norm2 = nn.BatchNorm1d(Config.n_embd)
        self.attn = Attention()
        self.mlp = nn.Sequential(
            nn.Linear(Config.n_embd,  Config.n_embd // 16),
            nn.Linear(Config.n_embd // 16, Config.n_embd),
            nn.Dropout(Config.resid_pdrop),
        )

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):  # <-- исправляем здесь
        super(LanguageModel, self).__init__()  # <-- исправляем здесь

        self.tok_emb = nn.Embedding(Config.vocab_size, Config.n_embd)  # <-- исправляем здесь
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(Config.n_layer)])
        self.norm_f = nn.BatchNorm1d(Config.n_embd)
        self.head = nn.Linear(Config.n_embd, Config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def forward(self, idx, targets):
        b, t = idx.size()
        assert t <= Config.seq_len, "Cannot forward, model block size is exhausted."

        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction='mean') #поменял на mean чтобы выдавался общий лосс
        return logits, loss

lm = LanguageModel()
vocab = ["мама", "компьютер", "мыла", "раму", "текст", "сгенерировал", "длинный"]  # <-- исправляем здесь
batch = ["Мама мыла раму", "Компьютер сгенерировал длинный текст"]  # <-- исправляем здесь

PAD = 0 #инициализирую нужные переменные 
BOS = 1
EOS = 2
UNK = 3
token2idx = {token: idx for idx, token in enumerate(vocab)}

batch = [[token2idx.get(token.text, 3) for token in razdel.tokenize(text.lower())] for text in batch]
MAX_LEN = 32
batch_input = torch.zeros((len(batch), MAX_LEN), dtype=torch.long)
# padleft each sample with PAD to MAX_LEN
for i, row in enumerate(batch):
    row = torch.tensor(row)
    batch_input[i, -len(row) - 2] = BOS
    batch_input[i, -len(row) - 1:-1] = row
    batch_input[i, -1] = EOS

logits, loss = lm.forward(batch_input, batch_input)
print(loss.shape)
loss.mean().backward()
print(loss)