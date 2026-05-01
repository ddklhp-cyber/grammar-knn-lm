"""
Quick test: runs inference on sample queries using the trained model + kNN datastore.

Usage:
    python test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference.knn_lm import (
    load_config, TinyLLM, Tokenizer,
    get_hidden_states, knn_search, search_docs, generate, can_word_follow,
    CHECKPOINT, VOCAB_PATH, DATASTORE_PATH, WORD_CONNS_PATH
)
import torch
import json


def main():
    if not CHECKPOINT.exists():
        print("ERROR: No trained model found.")
        print("Run:  python -m training.train_grammar")
        return

    if not DATASTORE_PATH.exists():
        print("ERROR: No datastore found.")
        print("Run:  python -m inference.knn_lm --build")
        return

    print("Loading model...")
    cfg = load_config(str(VOCAB_PATH))
    model = TinyLLM(cfg)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
    model.eval()
    tokenizer = Tokenizer(str(VOCAB_PATH))

    data = torch.load(DATASTORE_PATH, map_location='cpu')
    keys, vals = data['keys'], data['vals']

    word_conns = None
    if WORD_CONNS_PATH.exists():
        with open(WORD_CONNS_PATH) as f:
            word_conns = json.load(f)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Datastore: {len(vals)} entries")
    if word_conns:
        print(f"Grammar rules: {len(word_conns)} words")
    print()

    # Test queries
    queries = [
        "What is water made of",
        "What is the sun",
        "How do birds fly",
        "What is a computer",
        "Why is the sky blue",
    ]

    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for query in queries:
        results = search_docs(query)
        if not results:
            print(f"\nQ: {query}")
            print(f"A: [no search results]")
            continue

        output = generate(model, tokenizer, keys, vals, query, results, word_conns)
        print(f"\nQ: {query}")
        print(f"   Search: {results[0][:70]}...")
        print(f"A: {output}")

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == '__main__':
    main()
