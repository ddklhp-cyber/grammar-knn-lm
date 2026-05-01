"""
Link-grammar engine using rules extracted from CMU's full 4.0.dict.
691 connector entries (vs 9 in our original, 61 in tiny.dict).

Connector matching: word A can link to word B if A has X+ and B has X-.
Base letter determines link type (S=subject, O=object, D=determiner, etc.)
"""
import json
import re
from pathlib import Path

TINY_DICT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'tiny.dict'
FULL_RULES_PATH = Path(__file__).resolve().parents[1] / 'data' / 'link_grammar_full.json'

CONNECTOR_RE = re.compile(r'[A-Z][A-Za-z*]*[+-]')

_pos_left = None
_pos_right = None


def _ensure_loaded():
    global _pos_left, _pos_right
    if _pos_left is not None:
        return

    _pos_left = {}
    _pos_right = {}
    categories = ['noun', 'verb', 'adjective', 'adverb', 'pronoun',
                  'preposition', 'conjunction', 'determiner', 'other']

    if FULL_RULES_PATH.exists():
        # Use full rules for embedding init (more info)
        # But for generation filtering, use tiny.dict (stricter)
        pass

    if TINY_DICT_PATH.exists():
        word_rules = parse_tiny_dict(TINY_DICT_PATH)
        from tokenizer import guess_pos
        for pos in categories:
            _pos_left[pos] = set()
            _pos_right[pos] = set()
        for word, (left, right) in word_rules.items():
            pos = guess_pos(word)
            if pos in _pos_left:
                _pos_left[pos] |= left
                _pos_right[pos] |= right


def parse_tiny_dict(path):
    """Parse tiny.dict into {word: (left_connectors, right_connectors)}."""
    word_rules = {}
    text = path.read_text()
    lines = []
    for line in text.split('\n'):
        line = line.split('%')[0].strip()
        if line:
            lines.append(line)
    content = ' '.join(lines)

    entries = content.split(';')
    for entry in entries:
        entry = entry.strip()
        if ':' not in entry:
            continue
        words_part, expr = entry.split(':', 1)
        words = [w.strip().lower().replace('.v', '').replace('.p', '').replace('.a', '')
                 for w in re.split(r'\s+', words_part.strip())
                 if w.strip() and not w.startswith('#') and not w.startswith('<')]
        left = set(c[:-1] for c in CONNECTOR_RE.findall(expr) if c.endswith('-'))
        right = set(c[:-1] for c in CONNECTOR_RE.findall(expr) if c.endswith('+'))
        for w in words:
            if w and w[0].isalpha():
                if w in word_rules:
                    word_rules[w] = (word_rules[w][0] | left, word_rules[w][1] | right)
                else:
                    word_rules[w] = (left, right)
    return word_rules


def can_connect(left_pos, right_pos):
    """Check if right_pos can follow left_pos via link-grammar connectors.
    Left offers X+ connectors, right needs X- connectors.
    Match if left has 'Foo+' and right has 'Foo-' (same base)."""
    _ensure_loaded()
    if left_pos is None:
        return right_pos in ('determiner', 'pronoun', 'noun', 'adverb', 'adjective', 'other')

    # left_offers: connectors with + stripped (what left sends rightward)
    # right_needs: connectors with - stripped (what right accepts from left)
    left_offers = _pos_right.get(left_pos, set())  # these had + in original
    right_needs = _pos_left.get(right_pos, set())   # these had - in original

    # Direct match
    if left_offers & right_needs:
        return True

    # Prefix match: "S" offers can satisfy "Ss" need (or vice versa)
    for lo in left_offers:
        for rn in right_needs:
            if lo.startswith(rn) or rn.startswith(lo):
                return True

    if right_pos == 'conjunction':
        return left_pos in ('noun', 'verb', 'adjective', 'adverb', 'pronoun')

    return False


def get_allowed_pos(prev_pos):
    """Return list of POS categories that can follow prev_pos."""
    _ensure_loaded()
    categories = ['noun', 'verb', 'adjective', 'adverb', 'pronoun',
                  'preposition', 'conjunction', 'determiner', 'other']
    return [cat for cat in categories if can_connect(prev_pos, cat)]
