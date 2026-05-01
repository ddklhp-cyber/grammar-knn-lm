"""
kNN-LM: Build datastore + inference using grammar model hidden states.

Usage:
    python -m inference.knn_lm --build    # Build datastore
    python -m inference.knn_lm            # Interactive inference
"""
import argparse
import json
import re
import glob
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import TinyLLM, load_config
from model.tokenizer import Tokenizer

CHECKPOINT = ROOT / 'checkpoints' / 'model.pt'
VOCAB_PATH = ROOT / 'checkpoints' / 'vocab.json'
KNOWLEDGE_DIR = ROOT / 'data' / 'knowledge'
DATASTORE_PATH = ROOT / 'checkpoints' / 'datastore.pt'
WORD_CONNS_PATH = ROOT / 'data' / 'word_connectors.json'


def get_hidden_states(model, input_ids):
    model.eval()
    with torch.no_grad():
        B, T = input_ids.shape
        positions = torch.arange(T).unsqueeze(0).expand(B, -1)
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        hidden = model.embed_tokens(input_ids)
        for layer in model.layers:
            hidden = layer(hidden, positions, mask)
        hidden = model.norm(hidden)
    return hidden


def build_datastore(model, tokenizer):
    print("Building datastore from knowledge docs...")
    all_keys, all_vals = [], []

    for fpath in sorted(KNOWLEDGE_DIR.glob('**/*.txt')):
        text = fpath.read_text(encoding='utf-8', errors='ignore')
        for sent in re.split(r'<\|end\|>|\n|(?<=[.!?])\s+', text):
            sent = re.sub(r'<\|[^|]+\|>', '', sent).strip()
            if len(sent) < 10:
                continue
            ids = tokenizer.encode(sent)
            if len(ids) < 3:
                continue
            input_ids = torch.tensor([ids])
            hidden = get_hidden_states(model, input_ids)
            for i in range(len(ids) - 1):
                all_keys.append(hidden[0, i])
                all_vals.append(ids[i + 1])

    keys = torch.stack(all_keys)
    vals = torch.tensor(all_vals)
    DATASTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'keys': keys, 'vals': vals}, DATASTORE_PATH)
    print(f"Datastore: {len(vals)} entries, {keys.shape[1]} dims, {keys.element_size()*keys.nelement()/1024/1024:.1f} MB")


def knn_search(query_vec, keys, vals, vocab_size, k=16):
    dists = torch.cdist(query_vec.unsqueeze(0), keys).squeeze(0)
    topk_dists, topk_idx = dists.topk(k, largest=False)
    topk_probs = F.softmax(-topk_dists, dim=0)
    token_probs = torch.zeros(vocab_size)
    for prob, idx in zip(topk_probs, topk_idx):
        token_probs[vals[idx]] += prob
    return token_probs


def search_docs(query, top_k=2):
    stop_words = {'what', 'is', 'a', 'an', 'the', 'how', 'do', 'does', 'why', 'when', 'where', 'who', 'which', 'are', 'was', 'were', 'of', 'in', 'on', 'to', 'for', 'it', 'that', 'this'}
    query_words = set(w.lower() for w in query.split() if len(w) > 1)
    query_content = query_words - stop_words
    results = []
    for fpath in glob.glob(str(KNOWLEDGE_DIR / '**/*.txt'), recursive=True):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        for sent in re.split(r'<\|end\|>|\n', text):
            sent = sent.strip()
            if len(sent) < 5:
                continue
            clean = re.sub(r'<\|[^|]+\|>', '', sent).strip()
            sent_words = set(w.lower() for w in clean.split())
            score = len(query_content & sent_words)
            if score > 0:
                results.append((score, clean))
    results.sort(key=lambda x: -x[0])
    return [s for _, s in results[:top_k]]


def can_word_follow(prev_word, next_word, word_conns):
    """Word-level link-grammar check."""
    if prev_word is None or word_conns is None:
        return True
    prev_entry = word_conns.get(prev_word.lower())
    next_entry = word_conns.get(next_word.lower())
    if prev_entry is None or next_entry is None:
        return True
    for po in prev_entry['right']:
        for nn in next_entry['left']:
            if po.startswith(nn) or nn.startswith(po):
                return True
    return False


def generate(model, tokenizer, keys, vals, query, search_results, word_conns=None, max_tokens=25, lam=0.5):
    model.eval()
    candidates = set()
    for sent in search_results:
        for w in sent.split():
            w = re.sub(r'[^a-zA-Z]', '', w).lower()
            if len(w) > 1 and w in tokenizer.vocab:
                candidates.add(tokenizer.vocab[w])
    candidates.add(tokenizer.special_id('<|end|>'))
    candidate_list = list(candidates)

    prompt_ids = tokenizer.encode(query)
    input_ids = torch.tensor([prompt_ids])
    output_ids = []
    end_id = tokenizer.special_id('<|end|>')
    used = set()

    with torch.no_grad():
        for _ in range(max_tokens):
            hidden = get_hidden_states(model, input_ids)
            last_hidden = hidden[0, -1]
            model_logits = model.lm_head(last_hidden)
            model_probs = F.softmax(model_logits, dim=-1)
            knn_probs = knn_search(last_hidden, keys, vals, model.config.vocab_size)
            combined = lam * knn_probs + (1 - lam) * model_probs

            # Score candidates with grammar filter
            prev_word = tokenizer.reverse_vocab.get(output_ids[-1], '') if output_ids else None
            scores = torch.full_like(combined, float('-inf'))
            for cid in candidate_list:
                if cid in used and cid != end_id:
                    continue
                next_word = tokenizer.reverse_vocab.get(cid, '')
                if not can_word_follow(prev_word, next_word, word_conns):
                    scores[cid] = combined[cid] - 3.0  # soft penalty
                else:
                    scores[cid] = combined[cid]

            next_id = scores.argmax().item()
            if next_id == end_id:
                break
            output_ids.append(next_id)
            used.add(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    return tokenizer.decode(output_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Build datastore')
    args = parser.parse_args()

    print("Loading model...")
    cfg = load_config(str(VOCAB_PATH))
    model = TinyLLM(cfg)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
    model.eval()
    tokenizer = Tokenizer(str(VOCAB_PATH))

    if args.build:
        build_datastore(model, tokenizer)
        return

    if not DATASTORE_PATH.exists():
        print("No datastore. Run: python -m inference.knn_lm --build")
        return

    data = torch.load(DATASTORE_PATH, map_location='cpu')
    keys, vals = data['keys'], data['vals']
    print(f"Datastore: {len(vals)} entries")

    word_conns = None
    if WORD_CONNS_PATH.exists():
        with open(WORD_CONNS_PATH) as f:
            word_conns = json.load(f)
        print(f"Grammar rules: {len(word_conns)} words")

    print("Ready.\n")
    while True:
        query = input("> ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue
        results = search_docs(query)
        if not results:
            print("No matches.\n")
            continue
        print(f"\n[Search: {len(results)} matches]")
        for i, s in enumerate(results[:2], 1):
            print(f"  {i}. {s[:80]}")
        output = generate(model, tokenizer, keys, vals, query, results, word_conns)
        print(f"\n[Answer] {output}\n")


if __name__ == '__main__':
    main()
