#!/usr/bin/env python3
"""Download GloVe 6B 50d, take top-10k words, write compact binary asset.

Binary format:
  [u32 LE: word_count][u32 LE: dim=50]
  for each word (alphabetical order):
    [u8: word_len][bytes: word (utf-8)][f32 LE × 50: vector]
"""
import struct
import urllib.request
import zipfile
import io
from pathlib import Path

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
TOP_N = 10_000
DIM = 50
ROOT = Path(__file__).parent.parent  # project root
OUT_FILE = ROOT / "assets" / "glove_10k_50d.bin"


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading GloVe 6B (822 MB)...")
    with urllib.request.urlopen(GLOVE_URL, timeout=300) as response:
        data = response.read()

    print("Extracting glove.6B.50d.txt...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("glove.6B.50d.txt") as f:
            lines = f.read().decode("utf-8").splitlines()

    # Take top N (file is frequency-sorted; first line = most common word)
    entries = []
    for line in lines[:TOP_N]:
        parts = line.split()
        word = parts[0]
        vec = [float(x) for x in parts[1:]]
        if len(vec) == DIM:
            entries.append((word, vec))

    # Sort alphabetically so Rust can binary-search
    entries.sort(key=lambda e: e[0])

    print(f"Writing {len(entries)} words to {OUT_FILE}...")
    with open(OUT_FILE, "wb") as f:
        f.write(struct.pack("<II", len(entries), DIM))
        for word, vec in entries:
            wb = word.encode("utf-8")
            f.write(struct.pack(f"<B{len(wb)}s", len(wb), wb))
            f.write(struct.pack(f"<{DIM}f", *vec))

    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"Done. {OUT_FILE}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
