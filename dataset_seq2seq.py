"""
dataset_seq2seq.py — PyTorch Dataset for Piano Score Rearrangement

Loads pairs.jsonl, encodes tokens to integer IDs using vocab.json,
and applies optional pitch transposition augmentation (±2 semitones).

Sequence layout (matching paper Fig. 2b):
  Encoder input : [Dsrc, Dtgt, bar, ...tokens..., <eos>]
  Decoder input : [<sos>, Dtgt, bar, ...tokens...]        ← teacher-forced
  Decoder target: [Dtgt, bar, ...tokens..., <eos>]        ← loss target

Usage:
    dataset = ScorePairDataset('data/pairs.jsonl', 'data/vocab.json', augment=True)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True,
                         collate_fn=make_collate_fn(dataset.pad_id))
"""

import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Pitch transposition helpers
# ---------------------------------------------------------------------------

# Semitones above C for each diatonic step
_STEP_TO_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# Accidental → semitone offset
_ALTER_TO_OFFSET = {'': 0, '#': 1, '##': 2, 'b': -1, 'bb': -2}

# Chromatic note names indexed by pitch class [0..11]
_PC_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_PC_FLAT  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Circle of fifths: fifths value (-7..+7) → root pitch class
_FIFTHS_TO_PC = {
    -7: 11, -6: 6, -5: 1, -4: 8, -3: 3, -2: 10, -1: 5,
     0:  0,  1: 7,  2: 2,  3: 9,  4: 4,  5: 11,  6: 6,  7: 1,
}

# Root pitch class → preferred fifths (fewest accidentals, within [-7, +7])
_PC_TO_FIFTHS = {
    0:  0,   # C
    1: -5,   # Db  (5 flats — prefer over C# 7 sharps)
    2:  2,   # D
    3: -3,   # Eb
    4:  4,   # E
    5: -1,   # F
    6:  6,   # F#  (6 sharps — prefer over Gb 6 flats; tie broken by convention)
    7:  1,   # G
    8: -4,   # Ab
    9:  3,   # A
    10: -2,  # Bb
    11:  5,  # B   (5 sharps — prefer over Cb 7 flats)
}


def _parse_note_name(name: str) -> tuple[str, str, int]:
    """
    Parse a note name string (no 'note_' prefix) into (step, alter, octave).
    Examples:  'C4' → ('C', '', 4)
               'F#5' → ('F', '#', 5)
               'Bb3' → ('B', 'b', 3)
               'A##2' → ('A', '##', 2)
               'Dbb-1' → ('D', 'bb', -1)
    """
    step = name[0]
    rest = name[1:]
    if rest.startswith('##'):
        alter, rest = '##', rest[2:]
    elif rest.startswith('bb'):
        alter, rest = 'bb', rest[2:]
    elif rest.startswith('#'):
        alter, rest = '#', rest[1:]
    elif rest.startswith('b'):
        alter, rest = 'b', rest[1:]
    else:
        alter = ''
    return step, alter, int(rest)


def _parse_plain_pitch(name: str) -> tuple[str, str]:
    """
    Parse a pitch-class name with no octave (used for bass/chord roots).
    Examples: 'F#' → ('F', '#'),  'Bb' → ('B', 'b'),  'Bbb' → ('B', 'bb')
    """
    step = name[0]
    rest = name[1:]
    if rest.startswith('##'):
        alter = '##'
    elif rest.startswith('bb'):
        alter = 'bb'
    elif rest.startswith('#'):
        alter = '#'
    elif rest.startswith('b'):
        alter = 'b'
    else:
        alter = ''
    return step, alter


def _plain_pitch_to_pc(name: str) -> int:
    step, alter = _parse_plain_pitch(name)
    return (_STEP_TO_SEMITONE[step] + _ALTER_TO_OFFSET[alter]) % 12


def _pc_to_name(pc: int, prefer_sharps: bool) -> str:
    return _PC_SHARP[pc] if prefer_sharps else _PC_FLAT[pc]


def _parse_key_fifths(token: str) -> int:
    """'key_flat_3' → -3,  'key_sharp_2' → 2,  'key_natural_0' → 0"""
    if token == 'key_natural_0':
        return 0
    parts = token.split('_')
    v = int(parts[2])
    return -v if parts[1] == 'flat' else v


def _fifths_to_key_token(fifths: int) -> str:
    if fifths == 0:
        return 'key_natural_0'
    return f'key_sharp_{fifths}' if fifths > 0 else f'key_flat_{abs(fifths)}'


def _parse_chord_root(token: str) -> tuple[str, str]:
    """
    Split a chord token into (root, chord_type).
    'chord_F#m7' → ('F#', 'm7'),  'chord_C' → ('C', '')
    """
    rest = token[6:]  # strip 'chord_'
    step = rest[0]
    rest = rest[1:]
    if rest.startswith('##'):
        alter, rest = '##', rest[2:]
    elif rest.startswith('bb'):
        alter, rest = 'bb', rest[2:]
    elif rest.startswith('#'):
        alter, rest = '#', rest[1:]
    elif rest.startswith('b'):
        alter, rest = 'b', rest[1:]
    else:
        alter = ''
    return step + alter, rest


def transpose_tokens(tokens: list[str], shift: int, prefer_sharps: bool | None = None) -> list[str]:
    """
    Transpose all pitch-bearing tokens by `shift` semitones.

    Affected token types:
      note_*  — full note with octave
      key_*   — key signature (circle-of-fifths shift)
      bass_*  — bass note pitch class
      chord_* — chord root pitch class (chord quality unchanged)

    All other tokens are passed through unchanged.

    prefer_sharps: spelling preference for chromatic pitches.
                   None = auto (True if shift > 0, False if shift < 0).
    """
    if shift == 0:
        return tokens

    if prefer_sharps is None:
        prefer_sharps = shift > 0

    result = []
    for tok in tokens:
        if tok.startswith('note_'):
            step, alter, octave = _parse_note_name(tok[5:])
            midi = 12 * (octave + 1) + _STEP_TO_SEMITONE[step] + _ALTER_TO_OFFSET[alter]
            new_midi = midi + shift
            new_pc   = new_midi % 12
            new_oct  = new_midi // 12 - 1
            result.append(f'note_{_pc_to_name(new_pc, prefer_sharps)}{new_oct}')

        elif tok.startswith('key_'):
            fifths  = _parse_key_fifths(tok)
            pc      = _FIFTHS_TO_PC[fifths]
            new_pc  = (pc + shift) % 12
            result.append(_fifths_to_key_token(_PC_TO_FIFTHS[new_pc]))

        elif tok.startswith('bass_'):
            pc     = _plain_pitch_to_pc(tok[5:])
            new_pc = (pc + shift) % 12
            result.append(f'bass_{_pc_to_name(new_pc, prefer_sharps)}')

        elif tok.startswith('chord_'):
            root, chord_type = _parse_chord_root(tok)
            pc     = _plain_pitch_to_pc(root)
            new_pc = (pc + shift) % 12
            result.append(f'chord_{_pc_to_name(new_pc, prefer_sharps)}{chord_type}')

        else:
            result.append(tok)

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScorePairDataset(Dataset):
    """
    PyTorch Dataset over piano score rearrangement pairs.

    Each item returns (src_ids, tgt_ids) as 1-D LongTensors.

    Encoder input  src_ids: [Dsrc, Dtgt, tok…, <eos>]
    Decoder full   tgt_ids: [<sos>, Dtgt, tok…, <eos>]
    → At training time, split by collate_fn into:
        decoder input  = tgt_ids[:-1]
        decoder target = tgt_ids[1:]

    Parameters
    ----------
    pairs_path    : path to pairs.jsonl
    vocab_path    : path to vocab.json
    augment       : if True, apply random pitch transposition each epoch
    aug_semitones : semitone offsets to sample from (default ±2, including 0)
    max_src_len   : hard truncation for encoder sequence
    max_tgt_len   : hard truncation for full decoder sequence (before split)
    """

    def __init__(
        self,
        pairs_path: str,
        vocab_path: str,
        augment: bool = True,
        aug_semitones: tuple[int, ...] = (-2, -1, 0, 1, 2),
        max_src_len: int = 2048,
        max_tgt_len: int = 2048,
    ):
        with open(vocab_path, encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.token_to_id: dict[str, int] = vocab_data['token_to_id']
        self.pad_id = self.token_to_id['<pad>']
        self.sos_id = self.token_to_id['<sos>']
        self.eos_id = self.token_to_id['<eos>']
        self.unk_id = self.token_to_id.get('<unk>', 3)

        self.augment       = augment
        self.aug_semitones = list(aug_semitones)
        self.max_src_len   = max_src_len
        self.max_tgt_len   = max_tgt_len

        with open(pairs_path, encoding='utf-8') as f:
            self.pairs = [json.loads(line) for line in f if line.strip()]

    def _encode(self, tokens: list[str]) -> list[int]:
        """Convert token strings to IDs, using <unk> for out-of-vocab tokens."""
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        src_tokens: list[str] = pair['src_tokens']
        tgt_tokens: list[str] = pair['tgt_tokens']
        src_level:  str       = pair['src_level']   # e.g. 'Lv.2'
        tgt_level:  str       = pair['tgt_level']   # e.g. 'Lv.1'

        # Pitch transposition augmentation
        if self.augment:
            shift = random.choice(self.aug_semitones)
            if shift != 0:
                src_tokens = transpose_tokens(src_tokens, shift)
                tgt_tokens = transpose_tokens(tgt_tokens, shift)

        # Build sequences with difficulty conditioning tokens (paper Fig. 2b):
        #   src: [Dsrc, Dtgt, <content>, <eos>]
        #   tgt: [<sos>, Dtgt, <content>, <eos>]
        src_ids = self._encode([src_level, tgt_level] + src_tokens + ['<eos>'])
        tgt_ids = self._encode(['<sos>', tgt_level]   + tgt_tokens + ['<eos>'])

        # Truncate (rare, only for very long segments)
        src_ids = src_ids[:self.max_src_len]
        tgt_ids = tgt_ids[:self.max_tgt_len]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def make_collate_fn(pad_id: int):
    """
    Factory that returns a collate_fn configured with the given padding index.

    The returned function pads a list of (src, tgt) pairs into batch tensors:

      src         : (B, S)      — padded encoder input
      tgt_in      : (B, T-1)   — padded decoder input  (all but <eos>)
      tgt_out     : (B, T-1)   — padded decoder target (all but <sos>)
      src_pad_mask: (B, S)  bool — True at padding positions
      tgt_pad_mask: (B, T-1) bool

    Training usage:
        logits = model(src, tgt_in, src_pad_mask, tgt_pad_mask)
        loss   = F.cross_entropy(logits.view(-1, V), tgt_out.view(-1),
                                 ignore_index=pad_id)
    """
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        src_list, tgt_list = zip(*batch)

        src = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
        tgt = pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)

        tgt_in  = tgt[:, :-1]   # [<sos>, Dtgt, tok1, …, tokN]
        tgt_out = tgt[:, 1:]    # [Dtgt,  tok1, …, tokN, <eos>]

        src_pad_mask = (src == pad_id)
        tgt_pad_mask = (tgt_in == pad_id)

        return src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask

    return collate_fn


# ---------------------------------------------------------------------------
# Utility: train/val split
# ---------------------------------------------------------------------------

def make_splits(
    pairs_path: str,
    vocab_path: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[ScorePairDataset, ScorePairDataset]:
    """
    Return (train_dataset, val_dataset) split by song name so no song
    appears in both splits.
    """
    import math
    from torch.utils.data import Subset

    # Load all pairs to get song names
    with open(pairs_path, encoding='utf-8') as f:
        all_pairs = [json.loads(line) for line in f if line.strip()]

    songs = list({p['song'] for p in all_pairs})
    rng   = random.Random(seed)
    rng.shuffle(songs)

    n_val  = max(1, math.ceil(len(songs) * val_ratio))
    val_songs  = set(songs[:n_val])
    train_songs = set(songs[n_val:])

    train_idx = [i for i, p in enumerate(all_pairs) if p['song'] in train_songs]
    val_idx   = [i for i, p in enumerate(all_pairs) if p['song'] in val_songs]

    full_train = ScorePairDataset(pairs_path, vocab_path, augment=True)
    full_val   = ScorePairDataset(pairs_path, vocab_path, augment=False)

    return Subset(full_train, train_idx), Subset(full_val, val_idx)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- transpose unit tests ---
    test_cases = [
        # (original, shift, expected)
        ('note_C4',   2,  'note_D4'),
        ('note_B4',   1,  'note_C5'),
        ('note_C4',  -1,  'note_B3'),
        ('note_F#5',  2,  'note_G#5'),
        ('key_natural_0',  2, 'key_sharp_2'),    # C → D (2 sharps)
        ('key_flat_1',    -2, 'key_flat_3'),     # F → Eb (3 flats)
        ('key_sharp_1',    1, 'key_flat_4'),     # G → Ab (4 flats)
        ('bass_C',    1, 'bass_C#'),
        ('bass_Bb',   2, 'bass_C'),
        ('chord_G',   2, 'chord_A'),
        ('chord_F#m', 1, 'chord_Gm'),
    ]
    print('Transpose unit tests:')
    passed = failed = 0
    for orig, shift, expected in test_cases:
        result = transpose_tokens([orig], shift)[0]
        ok = result == expected
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            failed += 1
            print(f'  [{status}] transpose({orig!r}, {shift:+d}) → {result!r}  (expected {expected!r})')
        else:
            passed += 1
    print(f'  {passed} passed, {failed} failed\n')

    # --- dataset loading ---
    dataset = ScorePairDataset('data/pairs.jsonl', 'data/vocab.json', augment=True)
    print(f'Dataset size    : {len(dataset):,}')

    src, tgt = dataset[0]
    print(f'Sample src shape: {src.shape}  ids[:8]: {src[:8].tolist()}')
    print(f'Sample tgt shape: {tgt.shape}  ids[:8]: {tgt[:8].tolist()}')

    # --- dataloader with collate ---
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        collate_fn=make_collate_fn(dataset.pad_id),
    )
    src_b, tgt_in, tgt_out, src_mask, tgt_mask = next(iter(loader))
    print(f'\nBatch shapes (B=4):')
    print(f'  src      : {tuple(src_b.shape)}')
    print(f'  tgt_in   : {tuple(tgt_in.shape)}')
    print(f'  tgt_out  : {tuple(tgt_out.shape)}')
    print(f'  src_mask : {tuple(src_mask.shape)}  (bool, padding = True)')
    print(f'  tgt_mask : {tuple(tgt_mask.shape)}')

    # --- train/val split ---
    train_ds, val_ds = make_splits('data/pairs.jsonl', 'data/vocab.json')
    print(f'\nTrain split     : {len(train_ds):,} pairs')
    print(f'Val split       : {len(val_ds):,} pairs')
