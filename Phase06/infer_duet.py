"""
infer_duet.py — Inference script for Phase 6 Duet Auto-Orchestration

Mirrors infer.py (Phase 4.1) with three changes for duet support (see spec §6.4):

  1. Conditioning token swap. The encoder source is prepended with
     `<task_melody>` (Dataset A) or `<task_duet>` (Dataset B) instead of
     Lv.src/Lv.tgt. The decoder is also forced to begin with the same task
     token via `init_token_idx` in `greedy_decode`.

  2. Multi-staff output for `--task duet`. The model emits a single flat
     token stream containing both `<track_violin>` and `<track_piano>`
     regions. After decoding we split at the markers, build separate
     music21 Parts (violin = treble single staff, piano = R/L PartStaffs),
     and combine them into one multi-staff Score.

  3. No Lv.* conditioning. Phase 6 trained on `<task_*>` only, so passing
     Lv.* tokens at inference would be out-of-distribution.

Usage:
    python Phase06/infer_duet.py --input piano.mxl --output duet.mxl \\
        --task duet --checkpoint data/checkpoints_duet/best.pt
    python Phase06/infer_duet.py --input piano.mxl --output melody.mxl \\
        --task melody --temperature 1.2 --top_k 10
"""

import argparse
import json
import os
import sys

# Make sibling modules in the repo root importable when this script is run
# from anywhere (`python Phase06/infer_duet.py` or `cd Phase06&& python infer_duet.py`).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from music21 import bar, instrument, layout, stream

from model import build_model
from score_to_tokens import MusicXML_to_tokens
from tokens_to_score import tokens_to_PartStaff, tokens_to_score
from build_pairs import split_into_bars, bars_to_tokens


# ---------------------------------------------------------------------------
# Task / token constants
# ---------------------------------------------------------------------------

VALID_TASKS = ('melody', 'duet')

TASK_TO_TOKEN = {
    'melody': '<task_melody>',
    'duet':   '<task_duet>',
}

TRACK_VIOLIN = '<track_violin>'
TRACK_PIANO  = '<track_piano>'

MAX_SRC_LEN = 2048   # must match PositionalEncoding max_len in model.py (Phase 6.3)

# Tokens that continue a note group (used by structural validators)
_NOTE_GROUP_CONTINUATIONS = {'stem', 'beam', 'tie', 'staccato', 'accent', 'tenuto'}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_segment(bar_tokens, task_token, token_to_id, eos_id):
    """
    Build the encoder input IDs for one segment:
        [task_token, task_token, <segment_tokens...>, <eos>]

    The task token is repeated twice to match the training-time layout
    (`ScorePairDataset.__getitem__` uses [src_cond, tgt_cond] for both
    piano-only and duet, where for duet src_cond == tgt_cond == task token).

    Tokens not present in the vocabulary are silently dropped.
    Truncated to MAX_SRC_LEN with the prefix and <eos> preserved.
    """
    task_id = token_to_id[task_token]
    ids = [task_id, task_id]
    for tok in bar_tokens:
        if tok in token_to_id:
            ids.append(token_to_id[tok])
    ids.append(eos_id)

    if len(ids) > MAX_SRC_LEN:
        ids = ids[:MAX_SRC_LEN - 1] + [eos_id]
    return ids


def ids_to_tokens(id_list, id_to_token):
    """Convert a list of integer token IDs to token strings, dropping unknowns."""
    return [id_to_token[i] for i in id_list if i in id_to_token]


# ---------------------------------------------------------------------------
# Structural validators
# ---------------------------------------------------------------------------

def _note_group_lengths_ok(tokens):
    """Every note/rest group must contain at least one `len_` token before the
    group ends. A note group can be a chord — multiple consecutive `note_*`
    tokens share the trailing `len_*`. A rest is a single-event group.

    Mirrors what `note_token_to_obj` requires to avoid an IndexError during
    detokenization. Differs from infer.py's `_bar_is_valid` by correctly
    permitting chord notation.
    """
    in_group = False   # are we currently inside a note/rest group?
    has_len  = False   # has the current group seen a len_ token?
    for t in tokens:
        prefix = t.split('_')[0]
        if prefix == 'note':
            if in_group and has_len:
                # End of previous group, start of a new note group
                in_group = True
                has_len  = False
            else:
                # Either starting fresh (in_group False) or chord continuation
                # (in_group True, has_len False) — both legal, just mark in-group
                in_group = True
        elif prefix == 'rest':
            # Rests cannot be chorded; the previous group must have a length
            if in_group and not has_len:
                return False
            in_group = True
            has_len  = False
        elif prefix == 'len':
            has_len = True
        elif prefix in _NOTE_GROUP_CONTINUATIONS or t in ('slur_start', 'slur_stop'):
            pass
        else:
            # A non-group token (bar, key, time, R, L, clef, …) closes any open group
            if in_group and not has_len:
                return False
            in_group = False
            has_len  = False
    return not (in_group and not has_len)


def _split_bars(tokens):
    """Split flat token list at 'bar' markers; returns list of per-bar lists."""
    bars, cur = [], []
    for t in tokens:
        if t == 'bar':
            if cur:
                bars.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        bars.append(cur)
    return bars


def violin_segment_valid(tokens):
    """Validate a single-staff (violin) segment: at least one bar, every
    note/rest group has a length token. No R/L marker is expected."""
    bars = _split_bars(tokens)
    return bool(bars) and all(_note_group_lengths_ok(b) for b in bars)


def piano_segment_valid(tokens):
    """Validate a piano segment.

    The piano region of a Dataset B target may be either:
      * Two-staff R/L (the normal case), or
      * Single-staff (no R/L) — happens for silent/tacet piano parts and for
        some PDMX scores where the piano is notated on one staff only.

    Both are accepted as long as every note/rest group has a length token.
    """
    bars = _split_bars(tokens)
    if not bars:
        return False
    return all(_note_group_lengths_ok(b) for b in bars)


def split_duet_output(tokens):
    """
    Split a Dataset B decoded segment into (violin_tokens, piano_tokens).

    Expected layout: [<track_violin>, ...violin..., <track_piano>, ...piano...].
    Returns (None, None) if either marker is missing or ordering is wrong.
    """
    try:
        vi = tokens.index(TRACK_VIOLIN)
        pi = tokens.index(TRACK_PIANO)
    except ValueError:
        return None, None
    if vi >= pi:
        return None, None
    return tokens[vi + 1: pi], tokens[pi + 1:]


# ---------------------------------------------------------------------------
# Detokenization helpers
# ---------------------------------------------------------------------------

def violin_tokens_to_part(violin_tokens):
    """Build a single-staff violin music21 Part from a flat token stream.

    The duet violin stream has no R/L markers (it's a single track), so we
    call `tokens_to_PartStaff` directly — `tokens_to_score` would crash on
    the missing R marker.
    """
    part = tokens_to_PartStaff(violin_tokens, key_=0, start_voice=0, slur_number=1)
    part.insert(0, instrument.Violin())
    part.elements[-1].rightBarline = bar.Barline('regular')
    return part


def piano_tokens_to_score(piano_tokens):
    """Detokenize the piano region into a music21 Score.

    Two layouts handled:
      • Two-staff piano (`R …  L …`): pass through `tokens_to_score`, which
        produces a Score with both PartStaffs and a brace StaffGroup.
      • Single-staff piano (no R/L marker): use `tokens_to_PartStaff` directly
        and wrap a fresh Score around the single Part so downstream assembly
        can treat both layouts uniformly.
    """
    if 'R' in piano_tokens:
        return tokens_to_score(piano_tokens)

    part = tokens_to_PartStaff(piano_tokens, key_=0, start_voice=0, slur_number=1)
    part.elements[-1].rightBarline = bar.Barline('regular')
    s = stream.Score()
    s.append(part)
    return s


def build_melody_score(violin_tokens):
    """Wrap a single violin Part in a Score (used by --task melody)."""
    s = stream.Score()
    s.append(violin_tokens_to_part(violin_tokens))
    return s


def build_duet_score(violin_tokens, piano_tokens):
    """
    Assemble the final multi-staff Score: violin Part on top, piano staves
    below. Handles both two-staff (R+L) and single-staff piano output.
    `instrument.Violin()` / `instrument.Piano()` are inserted so MusicXML
    renderers (soundslice, MuseScore) play each part with the right sound.
    """
    violin_part = violin_tokens_to_part(violin_tokens)
    piano_score = piano_tokens_to_score(piano_tokens)
    piano_parts = list(piano_score.getElementsByClass(stream.PartStaff))
    for p in piano_parts:
        p.insert(0, instrument.Piano())

    out = stream.Score()
    if len(piano_parts) == 2:
        piano_group = layout.StaffGroup(piano_parts, symbol='brace', barTogether=True)
        out.append([violin_part, piano_group, *piano_parts])
    elif len(piano_parts) == 1:
        out.append([violin_part, piano_parts[0]])
    else:
        raise ValueError(
            f'Unexpected piano staff count from tokens_to_score: {len(piano_parts)}'
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Duet inference: piano solo → violin melody (--task melody) '
                    'or → piano+violin duet (--task duet).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',  required=True, help='Input piano-solo MXL or XML file')
    p.add_argument('--output', required=True, help='Output MXL file')
    p.add_argument(
        '--task', required=True, choices=VALID_TASKS,
        help='melody = Dataset A (violin only); duet = Dataset B (piano + violin)',
    )
    p.add_argument(
        '--checkpoint', default='data/checkpoints_duet/best.pt',
        help='Trained Phase 6 model checkpoint (.pt file)',
    )
    p.add_argument('--vocab',  default='data/vocab.json', help='Vocabulary file')
    p.add_argument(
        '--seg_len', type=int, default=8,
        help='Bars per inference segment (4–8 recommended)',
    )
    p.add_argument(
        '--max_decode_len', type=int, default=2048,
        help='Max tokens generated per segment (duet targets can reach ~1880)',
    )
    p.add_argument(
        '--temperature', type=float, default=1.2,
        help='Sampling temperature (>1 adds variety, 1.0 = near-greedy)',
    )
    p.add_argument(
        '--top_k', type=int, default=10,
        help='Top-k sampling (0 = greedy argmax)',
    )
    p.add_argument(
        '--device', default=None,
        help='Device override (e.g. cpu, cuda:0). Auto-detected if omitted.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.input  = args.input.strip()
    args.output = args.output.strip()

    # ── device ────────────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device \
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device          : {device}')

    # ── vocab ─────────────────────────────────────────────────────────────
    with open(args.vocab, encoding='utf-8') as f:
        vocab_data = json.load(f)
    token_to_id = vocab_data['token_to_id']
    id_to_token = {v: k for k, v in token_to_id.items()}

    vocab_size = len(token_to_id)
    pad_id     = token_to_id['<pad>']
    sos_id     = token_to_id['<sos>']
    eos_id     = token_to_id['<eos>']

    task_token = TASK_TO_TOKEN[args.task]
    for required in (task_token, TRACK_VIOLIN, TRACK_PIANO):
        if required not in token_to_id:
            print(f'Error: required token "{required}" missing from vocab.', file=sys.stderr)
            sys.exit(1)

    # ── model ─────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(vocab_size, pad_id).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Epoch {ckpt.get("epoch", "?"):>4}  val_loss={ckpt.get("val_loss", float("nan")):.4f}  '
          f'params={model.count_parameters():,}')

    # ── tokenize input ────────────────────────────────────────────────────
    print(f'\nTokenizing: {args.input}')
    try:
        tokens = MusicXML_to_tokens(args.input, bar_major=True, note_name=True)
    except Exception as e:
        print(f'Error tokenizing input: {e}', file=sys.stderr)
        sys.exit(1)

    bars = split_into_bars(tokens)
    if not bars:
        print('Error: no bars found in input score.', file=sys.stderr)
        sys.exit(1)
    print(f'  Bars            : {len(bars)}')
    print(f'  Task            : {args.task} ({task_token})')

    # ── segment → model → collect outputs ─────────────────────────────────
    seg_len = max(4, min(args.seg_len, 8))
    starts  = list(range(0, len(bars), seg_len))
    n_segs  = len(starts)

    violin_stream = []   # concatenated violin tokens (both tasks contribute)
    piano_stream  = []   # concatenated piano tokens (only --task duet contributes)
    n_skipped     = 0    # segments dropped because output was malformed

    print(f'\nRunning model: {n_segs} segment(s), up to {seg_len} bars each')

    for seg_idx, start in enumerate(starts):
        seg_bars   = bars[start: start + seg_len]
        seg_tokens = bars_to_tokens(seg_bars)

        src_ids    = encode_segment(seg_tokens, task_token, token_to_id, eos_id)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        decoded_ids = model.greedy_decode(
            src_tensor,
            sos_id,
            eos_id,
            max_len=args.max_decode_len,
            init_token_idx=token_to_id[task_token],
            temperature=args.temperature,
            top_k=args.top_k,
        )[0]

        decoded = ids_to_tokens(decoded_ids, id_to_token)

        # init_token_idx forces task token as first output — strip it
        if decoded and decoded[0] == task_token:
            decoded = decoded[1:]

        if args.task == 'melody':
            if violin_segment_valid(decoded):
                violin_stream.extend(['bar'] + decoded if violin_stream else decoded)
            else:
                n_skipped += 1
        else:  # duet
            v, p = split_duet_output(decoded)
            if v is None or p is None:
                n_skipped += 1
            elif violin_segment_valid(v) and piano_segment_valid(p):
                violin_stream.extend(['bar'] + v if violin_stream else v)
                piano_stream.extend(['bar'] + p if piano_stream else p)
            else:
                n_skipped += 1

        if (seg_idx + 1) % 10 == 0 or (seg_idx + 1) == n_segs:
            extra = f'  (skipped so far: {n_skipped})' if n_skipped else ''
            print(f'  [{seg_idx + 1}/{n_segs}]  violin tokens: {len(violin_stream)}  '
                  f'piano tokens: {len(piano_stream)}{extra}')

    if n_skipped:
        print(f'  Note: {n_skipped}/{n_segs} segment(s) skipped (malformed model output).')

    if not violin_stream:
        print('Error: no valid violin output produced.', file=sys.stderr)
        sys.exit(1)
    if args.task == 'duet' and not piano_stream:
        print('Error: --task duet produced no valid piano output.', file=sys.stderr)
        sys.exit(1)

    # ── detokenize & write output ─────────────────────────────────────────
    print(f'\nDetokenizing: violin={len(violin_stream)} tokens'
          + (f', piano={len(piano_stream)} tokens' if args.task == 'duet' else ''))
    try:
        if args.task == 'melody':
            score = build_melody_score(violin_stream)
        else:
            score = build_duet_score(violin_stream, piano_stream)
    except Exception as e:
        print(f'Error detokenizing: {e}', file=sys.stderr)
        print('Try --temperature 1.0 --top_k 0 for stricter decoding.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    try:
        score.write('musicxml', fp=args.output)
    except Exception as e:
        print(f'Error writing output: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'Output written  : {args.output}')


if __name__ == '__main__':
    main()
