"""
Setup: download and build data files needed for training.

Usage:
    python scripts/setup_data.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
WORD_CONNS_PATH = DATA_DIR / 'word_connectors.json'


def build_word_connectors():
    """Build word_connectors.json from CMU link-grammar (55K+ English words)."""
    if WORD_CONNS_PATH.exists():
        print(f"word_connectors.json already exists")
        return

    lg_dir = ROOT / '_link-grammar'
    if not lg_dir.exists():
        print("Cloning link-grammar from GitHub...")
        subprocess.run(['git', 'clone', '--depth', '1',
                       'https://github.com/opencog/link-grammar.git', str(lg_dir)],
                      check=True)

    dict_path = lg_dir / 'data' / 'en' / '4.0.dict'
    words_dir = lg_dir / 'data' / 'en' / 'words'

    print("Parsing 4.0.dict...")
    text = dict_path.read_text()
    lines = [line.split('%')[0] for line in text.split('\n') if line.split('%')[0].strip()]
    content = ' '.join(lines)

    macros = {}
    for m in re.finditer(r'<([^>]+)>\s*:\s*([^;]+);', content):
        macros[m.group(1)] = m.group(2).strip()

    def expand(expr):
        for _ in range(5):
            for name, val in macros.items():
                expr = expr.replace(f'<{name}>', f' {val} ')
        return expr

    CONN_RE = re.compile(r'[A-Z][A-Za-z*]*[+-]')

    file_connectors = {}
    for m in re.finditer(r'/en/words/([^:]+):\s*([^;]+);', content):
        fname = m.group(1).strip()
        expr = expand(m.group(2))
        left = sorted(set(c[:-1] for c in CONN_RE.findall(expr) if c.endswith('-')))
        right = sorted(set(c[:-1] for c in CONN_RE.findall(expr) if c.endswith('+')))
        file_connectors[fname] = {'left': left, 'right': right}

    word_rules = {}
    for fname, conns in file_connectors.items():
        fpath = words_dir / fname
        if not fpath.exists():
            continue
        for line in fpath.read_text().split('\n'):
            w = line.strip()
            if not w or w.startswith('%'):
                continue
            w_clean = re.sub(r'\.[a-z]+$', '', w).lower()
            if w_clean and w_clean[0].isalpha():
                if w_clean in word_rules:
                    word_rules[w_clean] = {
                        'left': sorted(set(word_rules[w_clean]['left'] + conns['left'])),
                        'right': sorted(set(word_rules[w_clean]['right'] + conns['right']))
                    }
                else:
                    word_rules[w_clean] = dict(conns)

    for m in re.finditer(r'\n([\w.]+(?:\s+[\w.]+)*)\s*:\s*([^;]+);', text):
        expr = expand(m.group(2))
        left = sorted(set(c[:-1] for c in CONN_RE.findall(expr) if c.endswith('-')))
        right = sorted(set(c[:-1] for c in CONN_RE.findall(expr) if c.endswith('+')))
        for w in m.group(1).split():
            w_clean = re.sub(r'\.[a-z]+$', '', w).strip().lower()
            if w_clean and w_clean[0].isalpha() and len(w_clean) > 1:
                if w_clean in word_rules:
                    word_rules[w_clean] = {
                        'left': sorted(set(word_rules[w_clean]['left'] + left)),
                        'right': sorted(set(word_rules[w_clean]['right'] + right))
                    }
                else:
                    word_rules[w_clean] = {'left': left, 'right': right}

    with open(WORD_CONNS_PATH, 'w') as f:
        json.dump(word_rules, f)
    print(f"Built word_connectors.json: {len(word_rules)} words")


def download_training_data():
    """Download TinyStories for grammar training."""
    ts_path = DATA_DIR / 'tinystories.txt'
    if ts_path.exists():
        print("tinystories.txt already exists")
        return
    print("Note: For best results, add training text to data/ folder.")
    print("Recommended: TinyStories dataset (clean grammar, short sentences)")
    print("  https://huggingface.co/datasets/roneneldan/TinyStories")


if __name__ == '__main__':
    build_word_connectors()
    download_training_data()
    print("\nSetup complete! Next steps:")
    print("  1. Add training .txt files to data/")
    print("  2. python -m training.train_grammar")
    print("  3. python -m inference.knn_lm --build")
    print("  4. python test.py")
