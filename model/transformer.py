"""
Tiny transformer model (4 layers, ~300K params). Grammar-aware embedding init.
"""
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from model.tokenizer import GRAMMAR_CATEGORIES


class Config:
    num_layers = 6
    hidden_size = 64
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 8
    ffn_size = 128
    vocab_size = None
    max_seq_len = 128
    rope_theta = 10000.0
    tie_word_embeddings = True
    category_ranges = None
    word_subcategories = None


def load_config(vocab_path: str):
    cfg = Config()
    with open(vocab_path) as f:
        data = json.load(f)
    cfg.vocab_size = data['vocab_size']
    cfg.category_ranges = {k: tuple(v) for k, v in data['category_ranges'].items()}
    cfg.word_subcategories = data.get('word_subcategories', {})
    return cfg


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, positions):
        freqs = torch.outer(positions[0].float(), self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)
        cos = torch.cat([cos, cos], dim=-1).expand_as(x)
        sin = torch.cat([sin, sin], dim=-1).expand_as(x)
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_per_kv = config.num_q_heads // config.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_q_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_q_heads * config.head_dim, config.hidden_size, bias=False)
        self.rope = RoPE(config)

    def forward(self, hidden, positions, mask):
        B, T, _ = hidden.shape
        Q = self.q_proj(hidden).view(B, T, self.num_q_heads, self.head_dim)
        K = self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim)
        V = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim)
        Q, K = self.rope(Q, positions), self.rope(K, positions)
        K = K.repeat_interleave(self.q_per_kv, dim=2)
        V = V.repeat_interleave(self.q_per_kv, dim=2)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim) + mask
        output = torch.matmul(F.softmax(scores, dim=-1), V)
        return self.o_proj(output.transpose(1, 2).contiguous().view(B, T, -1))


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.ffn = FFN(config)

    def forward(self, hidden, positions, mask):
        hidden = hidden + self.attention(self.input_layernorm(hidden), positions, mask)
        hidden = hidden + self.ffn(self.post_attention_layernorm(hidden))
        return hidden


class TinyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._init_grammar_embeddings()
        print(f"TinyLLM: {sum(p.numel() for p in self.parameters()):,} params, vocab={config.vocab_size}")

    def _init_grammar_embeddings(self):
        """Initialize embeddings using word-level link-grammar connectors from 4.0.dict.
        Words with similar connector profiles start closer in embedding space."""
        if not self.config.category_ranges:
            return

        import json
        from model.tokenizer import GRAMMAR_CATEGORIES as CATS

        dim = self.config.hidden_size
        wc_path = Path(__file__).resolve().parents[1] / 'data' / 'word_connectors.json'

        # Load word-level connectors (3547 words from full 4.0.dict)
        word_conns = {}
        if wc_path.exists():
            with open(wc_path) as f:
                word_conns = json.load(f)

        # Build connector vocabulary: each unique connector -> a dimension
        all_connectors = set()
        for entry in word_conns.values():
            all_connectors |= {c + '-' for c in entry['left']}
            all_connectors |= {c + '+' for c in entry['right']}
        connector_list = sorted(all_connectors)
        conn_to_dim = {c: i % dim for i, c in enumerate(connector_list)}

        # Need reverse vocab: token_id -> word
        # Build from vocab in config (word_subcategories has id->subcat, but we need id->word)
        # We'll load vocab.json directly
        vocab_path = Path(__file__).resolve().parents[1] / 'checkpoints' / 'vocab.json'
        id_to_word = {}
        if vocab_path.exists():
            with open(vocab_path) as f:
                vdata = json.load(f)
            id_to_word = {v: k for k, v in vdata['vocab'].items()}

        num_cats = len(CATS)
        chunk = dim // num_cats

        with torch.no_grad():
            for i, cat in enumerate(CATS):
                if cat not in self.config.category_ranges:
                    continue
                start, end = self.config.category_ranges[cat]
                if end < start:
                    continue

                for idx in range(end - start + 1):
                    wid = start + idx
                    direction = torch.zeros(dim)
                    # Broad category signal
                    direction[i * chunk:(i + 1) * chunk] = 0.3

                    # Word-level connector signal
                    word = id_to_word.get(wid, '')
                    if word in word_conns:
                        entry = word_conns[word]
                        for c in entry['left']:
                            key = c + '-'
                            if key in conn_to_dim:
                                direction[conn_to_dim[key]] += 0.15
                        for c in entry['right']:
                            key = c + '+'
                            if key in conn_to_dim:
                                direction[conn_to_dim[key]] += 0.15

                    self.embed_tokens.weight[wid] = direction + torch.randn(dim) * 0.02

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=input_ids.device), diagonal=1)
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions, mask)
        return self.lm_head(self.norm(hidden))
