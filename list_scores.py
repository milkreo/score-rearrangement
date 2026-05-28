"""
list_scores.py — Generate a CSV of all scored piano arrangements with their
                 difficulty level and MXL path, sorted by difficulty.

Output: data/score_list.csv
Columns: level, song, mxl_path, token_path

Usage:
    python list_scores.py
    python list_scores.py --pairs data/pairs.jsonl --out data/score_list.csv
"""

import argparse
import csv
import json
import os


TOKENS_ROOT = 'tokens'
MXL_ROOT    = 'mxl'


def token_path_to_mxl_path(token_path):
    """Convert a token JSON path back to its original MXL path.

    tokens/1/11/Qm....json  →  mxl/1/11/Qm....mxl
    """
    # Normalise to forward slashes and make relative
    rel = token_path.replace('\\', '/')
    # Strip any leading absolute portion up to 'tokens/'
    idx = rel.find('tokens/')
    if idx != -1:
        rel = rel[idx + len('tokens/'):]   # e.g. 1/11/Qm....json
    rel = rel.replace('.json', '.mxl')
    return os.path.join(MXL_ROOT, rel)


def parse_args():
    p = argparse.ArgumentParser(
        description='List all training scores with difficulty levels.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--pairs', default='data/pairs.jsonl', help='pairs.jsonl file')
    p.add_argument('--out',   default='data/score_list.csv', help='output CSV path')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Reading {args.pairs} ...')

    # Collect unique (token_path, level, song) — one row per score
    seen = {}   # token_path → (level, song)
    with open(args.pairs, encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            path  = p['src_path']
            level = p['src_level']
            song  = p.get('song', '')
            if path not in seen:
                seen[path] = (level, song)

    print(f'Unique scores found: {len(seen)}')

    # Build rows
    rows = []
    for token_path, (level, song) in seen.items():
        mxl_path = token_path_to_mxl_path(token_path)
        rows.append({
            'level':      level,
            'song':       song,
            'mxl_path':   mxl_path,
            'token_path': token_path,
        })

    # Sort by level then song name
    level_order = {'Lv.1': 1, 'Lv.2': 2, 'Lv.3': 3, 'Lv.4': 4}
    rows.sort(key=lambda r: (level_order.get(r['level'], 9), r['song'].lower()))

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['level', 'song', 'mxl_path', 'token_path'])
        writer.writeheader()
        writer.writerows(rows)

    # Print level summary
    from collections import Counter
    counts = Counter(r['level'] for r in rows)
    print('\nLevel distribution:')
    for lv in ['Lv.1', 'Lv.2', 'Lv.3', 'Lv.4']:
        print(f'  {lv}: {counts.get(lv, 0):,} scores')
    print(f'\nOutput written: {args.out}')


if __name__ == '__main__':
    main()
