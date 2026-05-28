import os
import json
import glob

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# Scan piano-only tokens AND Phase 6 duet tokens. Either may be absent on a
# given machine; build_vocab() silently skips missing directories.
TOKENS_DIRS = [
    os.path.join(SCRIPT_DIR, "tokens"),                       # Phase 1.1
    os.path.join(SCRIPT_DIR, "Phase06", "tokens_duet"),       # Phase 6.1
]
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "vocab.json")

# ── Special tokens ────────────────────────────────────────────────────────────
# Order matters: padding must be index 0 so PyTorch's default ignore_index works.
# Phase 6.3 adds the last four — `<task_*>` is prepended to the source (analogous
# to `Lv.*`), `<track_*>` delimits per-track regions inside the Dataset B target.
SPECIAL_TOKENS = [
    "<pad>",            # 0  — pad sequences to the same length in a batch
    "<sos>",            # 1  — start-of-sequence (prepended to every target)
    "<eos>",            # 2  — end-of-sequence   (appended to every target)
    "<unk>",            # 3  — unknown token     (safety fallback)
    "Lv.1",             # 4  — difficulty conditioning (paper §3.1.2)
    "Lv.2",             # 5
    "Lv.3",             # 6
    "Lv.4",             # 7
    "<task_melody>",    # 8  — Phase 6.3: Dataset A conditioning
    "<task_duet>",      # 9  — Phase 6.3: Dataset B conditioning
    "<track_violin>",   # 10 — Phase 6.3: delimit violin region in Dataset B target
    "<track_piano>",    # 11 — Phase 6.3: delimit piano region in Dataset B target
]


def _iter_tokens_in_file(path: str):
    """Yield every token string in a token JSON file.

    Phase 1.1 (`tokens/*.json`) files are a flat list of strings.
    Phase 6.1 (`Phase06/tokens_duet/*.json`) files are dicts with
    `piano`/`violin` token streams.
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("piano", "violin"):
            stream = data.get(key)
            if isinstance(stream, list):
                yield from stream


def _collect_corpus_tokens(tokens_dirs: list[str]) -> tuple[set[str], int, int]:
    """Scan every JSON file under each existing tokens_dir, return
    (unique_tokens, files_ok, files_failed)."""
    unique: set[str] = set()
    files_ok = files_failed = 0

    for tdir in tokens_dirs:
        if not os.path.isdir(tdir):
            print(f"  Skipping missing dir: {tdir}")
            continue
        json_files = glob.glob(os.path.join(tdir, "**", "*.json"), recursive=True)
        print(f"  {tdir}: {len(json_files):,} token files")
        for i, path in enumerate(json_files):
            try:
                for tok in _iter_tokens_in_file(path):
                    unique.add(tok)
                files_ok += 1
            except Exception as e:
                files_failed += 1
                print(f"    Warning: could not read {path}: {e}")
            if (i + 1) % 10_000 == 0:
                print(f"    scanned {i + 1:,}/{len(json_files):,} files, "
                      f"{len(unique):,} unique tokens so far …")

    return unique, files_ok, files_failed


def build_vocab(tokens_dirs: list[str], output_path: str) -> None:
    """Build (or extend) vocab.json.

    Behaviour:
      * If `output_path` does not exist: build from scratch with SPECIAL_TOKENS
        first (fixed indices) followed by corpus tokens sorted alphabetically.
      * If it exists: PRESERVE every existing token→id mapping (so Phase 1–5
        checkpoints stay loadable) and append any missing tokens at the end —
        first any SPECIAL_TOKENS not yet present, then any new corpus tokens
        sorted alphabetically. This is how Phase 6.3 adds `<task_*>` /
        `<track_*>` plus the handful of new duet-only corpus tokens.
    """
    print(f"Token sources:")
    unique_tokens, files_ok, files_failed = _collect_corpus_tokens(tokens_dirs)
    print(f"\nFinished scanning. {files_ok:,} files OK, {files_failed} failed.")
    print(f"Unique corpus tokens found: {len(unique_tokens):,}")

    if not unique_tokens:
        print("ERROR: No tokens found in any source directory.")
        return

    # Don't double-count specials if they happen to be in the corpus.
    unique_tokens -= set(SPECIAL_TOKENS)

    existing = None
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            existing = json.load(f).get("token_to_id")

    if existing:
        # ── EXTEND mode: preserve existing IDs, append new tokens at the end ──
        vocab_list = sorted(existing, key=lambda t: existing[t])
        already = set(existing)

        appended_specials = [t for t in SPECIAL_TOKENS if t not in already]
        appended_corpus   = sorted(unique_tokens - already)

        vocab_list.extend(appended_specials)
        vocab_list.extend(appended_corpus)

        print(f"\nExtend mode (existing vocab found at {output_path}):")
        print(f"  Existing tokens preserved : {len(existing):,}")
        print(f"  New special tokens added  : {len(appended_specials)}  {appended_specials}")
        print(f"  New corpus tokens added   : {len(appended_corpus)}  "
              f"{appended_corpus[:10]}{' …' if len(appended_corpus) > 10 else ''}")
    else:
        # ── FRESH mode: specials first, then corpus sorted ───────────────────
        vocab_list = SPECIAL_TOKENS + sorted(unique_tokens)
        print(f"\nFresh build (no existing vocab):")
        print(f"  Special tokens : {len(SPECIAL_TOKENS)}")
        print(f"  Corpus tokens  : {len(unique_tokens):,}")

    token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}
    id_to_token = {str(idx): tok for idx, tok in enumerate(vocab_list)}

    print(f"\nFinal vocabulary size: {len(vocab_list):,} tokens")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved vocab to '{output_path}'")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\n── Sanity check ──────────────────────────────────────────────────")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:18s} → id {token_to_id[tok]}")


if __name__ == "__main__":
    build_vocab(TOKENS_DIRS, OUTPUT_PATH)
