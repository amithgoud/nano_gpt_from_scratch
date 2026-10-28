"""
Decoder-Only Transformer Language Model (nanoGPT style)
Implements Multi-Head Self-Attention, Feed-Forward Networks, LayerNorm, and Residual Connections.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """A single head of self-attention."""
    def __init__(self, n_emb, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size)
        self.query = nn.Linear(n_emb, head_size)
        self.val = nn.Linear(n_emb, head_size)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.n_emb = n_emb

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.val(x)    # (B, T, head_size)

        # Scaled dot-product attention
        # Note: scaling factor c**-0.5 where c = n_emb as per training configuration
        wei = q @ k.transpose(-2, -1) * (c ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out = wei @ v       # (B, T, head_size)
        return out


class MultiHeads(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, n_emb, num_heads, head_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.drop(out)
        return out


class FFn(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, n_emb, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: Communication followed by Computation with pre-LayerNorm."""
    def __init__(self, n_emb, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeads(n_emb, n_head, head_size, block_size, dropout)
        self.ffwn = FFn(n_emb, dropout)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwn(self.ln2(x))
        return x


# Preserve exact class names for backward compatibility with pickled state_dicts
head = Head
multiheads = MultiHeads
block = Block


class BigramLanguageModel(nn.Module):
    """
    Decoder-only Transformer Language Model.
    Architecture:
      - Token & Learned Positional Embeddings
      - Stack of N Transformer Blocks (Multi-Head Self-Attention + MLP)
      - Final LayerNorm + Linear LM Head
    """
    def __init__(
        self,
        vocab_size=116,
        n_emb=384,
        block_size=256,
        n_head=6,
        n_layer=6,
        dropout=0.2
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.pos_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[
            Block(n_emb=n_emb, n_head=n_head, block_size=block_size, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + Positional embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_emb)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=idx.device))  # (T, n_emb)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text auto-regressively from context tokens.
        idx: (B, T) tensor of integer token indices
        """
        for _ in range(max_new_tokens):
            con_idx = idx[:, -self.block_size:]
            logits, _ = self(con_idx)
            logits = logits[:, -1, :]  # focus on last step: (B, C)

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Alias for modern naming
GPTLanguageModel = BigramLanguageModel

