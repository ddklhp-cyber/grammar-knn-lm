"""
Sequential tokenizer: IDs grouped by grammar category.
Sub-categories from lexicon used for embedding initialization.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORD_PATTERN = re.compile(r"[A-Za-z']+")

GRAMMAR_CATEGORIES = [
    'noun', 'verb', 'adjective', 'adverb', 'pronoun',
    'preposition', 'conjunction', 'determiner', 'other',
]

SPECIAL_TOKEN_NAMES = [
    '<PAD>', '<BOS>', '<EOS>',
    '<|system|>', '<|user|>', '<|assistant|>', '<|end|>',
]

GRAMMAR_LEXICON: Dict[str, str] = {
    'i': 'pronoun', 'you': 'pronoun', 'he': 'pronoun', 'she': 'pronoun', 'it': 'pronoun',
    'we': 'pronoun', 'they': 'pronoun', 'me': 'pronoun', 'him': 'pronoun', 'her': 'pronoun',
    'a': 'determiner', 'an': 'determiner', 'the': 'determiner', 'this': 'determiner', 'that': 'determiner',
    'these': 'determiner', 'those': 'determiner', 'each': 'determiner', 'every': 'determiner',
    'in': 'preposition', 'on': 'preposition', 'at': 'preposition', 'from': 'preposition',
    'with': 'preposition', 'by': 'preposition', 'for': 'preposition', 'about': 'preposition',
    'into': 'preposition', 'through': 'preposition', 'during': 'preposition', 'before': 'preposition',
    'after': 'preposition', 'above': 'preposition', 'below': 'preposition', 'to': 'preposition',
    'and': 'conjunction', 'or': 'conjunction', 'but': 'conjunction', 'so': 'conjunction',
    'yet': 'conjunction', 'nor': 'conjunction', 'because': 'conjunction',
    'be': 'verb', 'have': 'verb', 'do': 'verb', 'say': 'verb', 'go': 'verb', 'can': 'verb',
    'get': 'verb', 'make': 'verb', 'know': 'verb', 'think': 'verb', 'take': 'verb', 'see': 'verb',
    'come': 'verb', 'want': 'verb', 'look': 'verb', 'use': 'verb', 'find': 'verb', 'give': 'verb',
    'run': 'verb', 'read': 'verb', 'write': 'verb', 'is': 'verb', 'are': 'verb', 'was': 'verb',
    'were': 'verb', 'has': 'verb', 'had': 'verb', 'does': 'verb', 'did': 'verb',
    'good': 'adjective', 'new': 'adjective', 'first': 'adjective', 'last': 'adjective',
    'large': 'adjective', 'old': 'adjective', 'big': 'adjective', 'high': 'adjective',
    'very': 'adverb', 'really': 'adverb', 'just': 'adverb', 'still': 'adverb',
    'never': 'adverb', 'always': 'adverb', 'often': 'adverb',
}

SUFFIX_POS_RULES = [
    ('ly', 'adverb'), ('ing', 'verb'), ('ed', 'verb'), ('es', 'verb'),
    ('ment', 'noun'), ('ness', 'noun'), ('tion', 'noun'), ('sion', 'noun'), ('ity', 'noun'),
    ('al', 'adjective'), ('able', 'adjective'), ('ible', 'adjective'),
    ('ous', 'adjective'), ('ful', 'adjective'), ('ive', 'adjective'), ('less', 'adjective'),
    ('er', 'noun'), ('or', 'noun'), ('ist', 'noun'), ('s', 'noun'),
]


def normalize_word(word: str) -> str:
    return word.lower().strip("'\"""''")


def guess_pos(word: str, pos_lexicon: Optional[Dict[str, str]] = None) -> str:
    w = normalize_word(word)
    if not w:
        return 'noun'
    if w in GRAMMAR_LEXICON:
        return GRAMMAR_LEXICON[w]
    if pos_lexicon and w in pos_lexicon:
        return pos_lexicon[w]
    for suffix, cat in SUFFIX_POS_RULES:
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            return cat
    return 'noun'


def load_pos_lexicon(path: str) -> Dict[str, str]:
    """Load lexicon, returns {word: broad_category}."""
    lexicon = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            w = normalize_word(entry.get('word', ''))
            pos = entry.get('pos', '')
            if pos.startswith('noun'): broad = 'noun'
            elif pos.startswith('verb'): broad = 'verb'
            elif pos.startswith('adj'): broad = 'adjective'
            elif pos.startswith('adv'): broad = 'adverb'
            else: broad = 'other'
            if w and broad in GRAMMAR_CATEGORIES:
                lexicon[w] = broad
    return lexicon


def load_sub_lexicon(path: str) -> Dict[str, str]:
    """Load lexicon, returns {word: full_sub_category} e.g. 'noun.location'."""
    lexicon = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            w = normalize_word(entry.get('word', ''))
            pos = entry.get('pos', '')
            if w and pos:
                lexicon[w] = pos
    return lexicon


def build_vocab(words_with_pos: List[Tuple[str, str]]) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int]]]:
    by_cat = {c: [] for c in GRAMMAR_CATEGORIES}
    seen = set()
    for word, pos in words_with_pos:
        norm = normalize_word(word)
        if norm in seen:
            continue
        seen.add(norm)
        cat = pos if pos in GRAMMAR_CATEGORIES else 'other'
        by_cat[cat].append(norm)

    vocab = {}
    category_ranges = {}
    current_id = 0
    for cat in GRAMMAR_CATEGORIES:
        start = current_id
        for word in by_cat[cat]:
            vocab[word] = current_id
            current_id += 1
        end = current_id - 1 if current_id > start else start
        category_ranges[cat] = (start, end)

    special_start = current_id
    for name in SPECIAL_TOKEN_NAMES:
        vocab[name] = current_id
        current_id += 1
    category_ranges['special'] = (special_start, current_id - 1)
    return vocab, category_ranges


_SPECIAL_ESCAPED = [re.escape(t) for t in sorted(SPECIAL_TOKEN_NAMES, key=len, reverse=True)]
TOKEN_PATTERN = re.compile('|'.join(_SPECIAL_ESCAPED) + r"|[A-Za-z']+")


class Tokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.category_ranges = {k: tuple(v) for k, v in data['category_ranges'].items()}
        self.vocab_size = data['vocab_size']
        self.word_subcategories = data.get('word_subcategories', {})
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        ids = []
        for tok in TOKEN_PATTERN.findall(text):
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                norm = normalize_word(tok)
                if norm in self.vocab:
                    ids.append(self.vocab[norm])
        return ids

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.reverse_vocab.get(i, f'<{i}>') for i in ids)

    def special_id(self, name: str) -> int:
        return self.vocab[name]


def save_vocab(vocab, category_ranges, path, word_subcategories=None):
    vocab_size = max(vocab.values()) + 1
    data = {'vocab': vocab, 'category_ranges': category_ranges, 'vocab_size': vocab_size}
    if word_subcategories:
        data['word_subcategories'] = word_subcategories
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Vocab saved: {len(vocab)} entries, vocab_size={vocab_size}")
