"""
Train the grammar model on full sentences.

Usage:
    python -m training.train_grammar
    python -m training.train_grammar --resume --steps 50000
"""
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tokenizer import (
    WORD_PATTERN, normalize_word, guess_pos, load_pos_lexicon, load_sub_lexicon,
    build_vocab, save_vocab, Tokenizer,
)
from model.transformer import TinyLLM, load_config

VOCAB_PATH = ROOT / 'checkpoints' / 'vocab.json'
CHECKPOINT = ROOT / 'checkpoints' / 'model.pt'
LEXICON = ROOT / 'data' / 'lexicon.jsonl'
DOCS_DIR = ROOT / 'data'

BATCH_SIZE = 16
MAX_LEN = 128
LR = 5e-4
STEPS = 100000
EVAL_INTERVAL = 2000


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=MAX_LEN):
        self.examples = []
        pad_id = tokenizer.special_id('<PAD>')
        bos_id = tokenizer.special_id('<BOS>')
        eos_id = tokenizer.special_id('<EOS>')
        for sent in sentences:
            ids = tokenizer.encode(sent)
            if len(ids) < 3:
                continue
            ids = [bos_id] + ids + [eos_id]
            if len(ids) > max_len:
                ids = ids[:max_len]
            length = len(ids)
            ids = ids + [pad_id] * (max_len - length)
            self.examples.append((torch.tensor(ids), length))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, length = self.examples[idx]
        x = tokens[:-1]
        y = tokens[1:].clone()
        y[length - 1:] = -100
        return x, y


def extract_sentences(docs_dir):
    sentences = []
    for f in sorted(docs_dir.glob('**/*.txt')):
        text = f.read_text(encoding='utf-8', errors='ignore')
        for sent in re.split(r'<\|end\|>|<\|user\|>|<\|assistant\|>|\n|(?<=[.!?])\s+', text):
            sent = re.sub(r'<\|[^|]+\|>', '', sent).strip()
            if 10 < len(sent) < 500:
                sentences.append(sent)
    return sentences


def prepare():
    pos_lexicon = load_pos_lexicon(str(LEXICON)) if LEXICON.exists() else {}
    texts = []
    for f in sorted(DOCS_DIR.glob('**/*.txt')):
        texts.append(f.read_text(encoding='utf-8', errors='ignore'))
    if not texts:
        print(f"No .txt files in {DOCS_DIR}")
        return False

    full_text = '\n'.join(texts)
    freq = Counter()
    for m in WORD_PATTERN.finditer(full_text):
        w = normalize_word(m.group())
        if w and len(w) > 1:
            freq[w] += 1

    top_words = [w for w, _ in freq.most_common(5000)]
    words_with_pos = [(w, guess_pos(w, pos_lexicon)) for w in top_words]
    vocab, category_ranges = build_vocab(words_with_pos)

    sub_lexicon = load_sub_lexicon(str(LEXICON)) if LEXICON.exists() else {}
    word_subcategories = {}
    for word, _ in words_with_pos:
        norm = normalize_word(word)
        if norm in vocab and norm in sub_lexicon:
            word_subcategories[str(vocab[norm])] = sub_lexicon[norm]

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, category_ranges, str(VOCAB_PATH), word_subcategories=word_subcategories)
    return True


def train(resume=False):
    cfg = load_config(str(VOCAB_PATH))
    tokenizer = Tokenizer(str(VOCAB_PATH))

    sentences = extract_sentences(DOCS_DIR)
    print(f"Extracted {len(sentences)} sentences")

    split = int(len(sentences) * 0.9)
    train_ds = SentenceDataset(sentences[:split], tokenizer)
    val_ds = SentenceDataset(sentences[split:], tokenizer)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TinyLLM(cfg)
    if resume and CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
        print(f"Resumed from {CHECKPOINT}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    best_val = float('inf')
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training: {STEPS} steps, batch={BATCH_SIZE}")
    t0 = time.time()
    train_iter = iter(train_loader)

    for step in range(1, STEPS + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        warmup = STEPS // 20
        if step < warmup:
            lr = LR * step / warmup
        else:
            decay = (step - warmup) / max(1, STEPS - warmup)
            lr = 1e-5 + (LR - 1e-5) * 0.5 * (1 + math.cos(math.pi * decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == 1:
            model.eval()
            vl = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= 10: break
                    vl.append(F.cross_entropy(model(vx).view(-1, cfg.vocab_size), vy.view(-1), ignore_index=-100).item())
            val_loss = sum(vl) / len(vl)
            saved = ''
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), CHECKPOINT)
                saved = ' *saved*'
            print(f"step {step:>6}/{STEPS} | lr {lr:.2e} | train {loss.item():.4f} | val {val_loss:.4f}{saved} | {time.time()-t0:.0f}s")

    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, default=STEPS)
    args = parser.parse_args()
    STEPS = args.steps

    if args.resume and CHECKPOINT.exists():
        print("=== Resuming training ===")
        train(resume=True)
    else:
        print("=== Preparing vocab ===")
        if prepare():
            print("\n=== Training ===")
            train()
