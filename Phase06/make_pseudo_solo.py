"""make_pseudo_solo.py -- Phase 6.4 helper.

Synthesize a **Pseudo Piano Solo** MXL from a real piano+violin duet in
`Phase06/tokens_duet/`, using the exact same merge logic as the training-data
builder (`build_pairs_duet.make_pseudo_bar`). The pseudo solo is the file you
feed to `infer_duet.py` as `--input`.

Alongside the pseudo solo, this script also writes the **ground-truth** scores
so you can compare what the model produces against the real thing:

    <outdir>/<name>_pseudo_solo.mxl   <- inference INPUT  (piano two-staff)
    <outdir>/<name>_gt_violin.mxl     <- ground truth for --task melody
    <outdir>/<name>_gt_duet.mxl       <- ground truth for --task duet

`build_pairs_duet.py` only emits token JSONL (training pairs); it never writes
a playable MXL. This script fills that gap for inference / demo / evaluation.

Usage:
    # by 0-based index into the sorted tokens_duet/*.json list
    python Phase06/make_pseudo_solo.py --index 0 --outdir output/pseudo

    # by song path relative to tokens_duet/
    python Phase06/make_pseudo_solo.py --song 0/0/Qma1c7....json --outdir output/pseudo
"""

import argparse
import json
import os
import sys
from collections import Counter
from glob import glob

# Make repo-root + Phase06 modules importable regardless of cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (REPO_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_pairs import split_into_bars, bars_to_tokens
from build_pairs_duet import make_pseudo_bar
from tokens_to_score import tokens_to_score
# Reuse the tested multi-staff / single-staff assembly from the inferencer so
# the ground-truth scores are built exactly like the model-output scores.
from infer_duet import build_melody_score, build_duet_score

TOKENS_DIR = os.path.join(SCRIPT_DIR, 'tokens_duet')


def resolve_song(args):
    """Return the absolute path to the chosen tokens_duet/*.json file."""
    if args.song:
        path = os.path.join(TOKENS_DIR, args.song.replace('\\', '/'))
        if not os.path.isfile(path):
            sys.exit(f'Error: song file not found: {path}')
        return path

    files = sorted(glob(os.path.join(TOKENS_DIR, '**', '*.json'), recursive=True))
    if not files:
        sys.exit(f'Error: no tokenized duets under {TOKENS_DIR}')
    if args.index < 0 or args.index >= len(files):
        sys.exit(f'Error: --index {args.index} out of range (0..{len(files) - 1})')
    return files[args.index]


def write_score(score, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    score.write('musicxml', fp=path)
    print(f'  wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--index', type=int, default=0,
                   help='0-based index into sorted tokens_duet/*.json (default: 0)')
    g.add_argument('--song', type=str, default=None,
                   help='Song path relative to tokens_duet/ (overrides --index)')
    ap.add_argument('--outdir', type=str, default='output/pseudo',
                    help='Directory for the generated MXL files')
    ap.add_argument('--name', type=str, default=None,
                    help='Output filename stem (default: derived from song id)')
    args = ap.parse_args()

    path = resolve_song(args)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    piano_tokens = d['piano']
    violin_tokens = d['violin']

    name = args.name or os.path.splitext(os.path.basename(path))[0]
    rel = os.path.relpath(path, TOKENS_DIR).replace('\\', '/')
    print(f'Source duet     : {rel}')

    # ── align bars and synthesize the pseudo solo (same as training builder) ──
    piano_bars = split_into_bars(piano_tokens)
    violin_bars = split_into_bars(violin_tokens)
    n_bars = min(len(piano_bars), len(violin_bars))
    piano_bars = piano_bars[:n_bars]
    violin_bars = violin_bars[:n_bars]
    print(f'Bars            : {n_bars} (piano {len(piano_tokens)} tok, violin {len(violin_tokens)} tok)')

    stats = Counter()
    pseudo_bars = [make_pseudo_bar(pb, vb, stats)
                   for pb, vb in zip(piano_bars, violin_bars)]
    if stats.get('skipped_multivoice_bars'):
        print(f'  multi-voice bars kept as-is : {stats["skipped_multivoice_bars"]}')
    if stats.get('bar_merge_errors'):
        print(f'  bars that fell back to piano: {stats["bar_merge_errors"]}')

    pseudo_tokens = bars_to_tokens(pseudo_bars)
    full_violin_tokens = bars_to_tokens(violin_bars)
    full_piano_tokens = bars_to_tokens(piano_bars)

    # ── write the three MXL files ─────────────────────────────────────────────
    # NOTE: the pseudo solo is written as *uncompressed* .xml because the
    # repo's loader (`score_to_tokens.load_MusicXML`) only finds a `.xml`
    # member inside an .mxl zip, whereas music21 names the inner file
    # `<name>.musicxml`. Writing .xml keeps the input readable by inference.
    # Ground-truth files stay .mxl (human comparison only, never re-parsed).
    print('\nWriting scores:')
    solo_path = os.path.join(args.outdir, f'{name}_pseudo_solo.xml')
    write_score(tokens_to_score(pseudo_tokens), solo_path)
    write_score(build_melody_score(full_violin_tokens),
                os.path.join(args.outdir, f'{name}_gt_violin.mxl'))
    write_score(build_duet_score(full_violin_tokens, full_piano_tokens),
                os.path.join(args.outdir, f'{name}_gt_duet.mxl'))

    print(f'\nNext: feed the pseudo solo to inference, e.g.\n'
          f'  .venv\\Scripts\\python.exe Phase06\\infer_duet.py '
          f'--input {solo_path} '
          f'--output {os.path.join(args.outdir, name + "_pred_duet.mxl")} --task duet')


if __name__ == '__main__':
    main()
