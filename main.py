import os
from score_to_tokens import MusicXML_to_tokens
from tokens_to_score import tokens_to_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MXL  = os.path.join(SCRIPT_DIR, "clementi-sonatina-no-1-op-36.mxl")

tokens = MusicXML_to_tokens(
    INPUT_MXL,
    bar_major=True,           # bar-major style (recommended)
    note_name=True,           # use note names like C4, not MIDI numbers
    tokenize_chord_symbols=True
)

s = tokens_to_score(tokens)
s.write('musicxml', 'output_score')  # saves output_score.musicxml