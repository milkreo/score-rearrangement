import os, json, csv, warnings
from music21 import musicxml
warnings.filterwarnings("ignore", category=musicxml.xmlToM21.MusicXMLWarning)
from score_to_tokens import MusicXML_to_tokens

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
csv_path    = os.path.join(SCRIPT_DIR, "PDMX.csv")
mxl_root    = os.path.join(SCRIPT_DIR, "mxl")
output_dir  = os.path.join(SCRIPT_DIR, "tokens")
os.makedirs(output_dir, exist_ok=True)

# collect piano-only mxl paths from metadata
piano_paths = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        tracks = row['tracks']
        if all(t == '0' for t in tracks.split('-')):
            # CSV paths look like ./mxl/1/11/Qm....mxl — make them absolute
            rel = row['mxl'].lstrip('./')
            piano_paths.append(os.path.join(os.path.dirname(csv_path), rel))

print(f"Found {len(piano_paths)} piano scores to tokenize.")

success, failed = 0, 0
for in_path in piano_paths:
    if not os.path.exists(in_path):
        continue  # mxl file not downloaded yet

    # mirror the mxl subdirectory structure under output_dir
    rel = os.path.relpath(in_path, mxl_root)
    out_path = os.path.join(output_dir, rel).replace('.mxl', '.json').replace('.xml', '.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        continue  # already processed

    try:
        tokens = MusicXML_to_tokens(in_path, bar_major=True, note_name=True, tokenize_chord_symbols=True)
        with open(out_path, 'w') as f:
            json.dump(tokens, f)
        success += 1
        if success % 500 == 0:
            print(f"Processed {success} files ({failed} failed)")
    except Exception as e:
        failed += 1
        print(f"Failed {in_path}: [{type(e).__name__}] {e}")

print(f"Done. {success} succeeded, {failed} failed.")
