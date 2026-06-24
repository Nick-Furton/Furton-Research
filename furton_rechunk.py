#!/usr/bin/env python3
"""
Furton Research — Re-Chunker (RAG fix)
======================================
The libraries were originally chunked at ~800 words / 80-word overlap. But the
embedding model (all-MiniLM-L6-v2) has max_seq_length = 256 tokens (~190 words),
so every embedding only reflected roughly the FIRST FIFTH of its chunk. Retrieval
was silently matching on truncated passages.

This script re-splits every chunk in each existing manifest into ~250-word
sub-chunks with ~40-word overlap, so each sub-chunk fits inside the model window.
It operates ONLY on the manifests already on disk — it does NOT re-collect any
sources.

What it does, per library:
  1. Backs up manifest.json  ->  manifest_backup.json   (only if no backup yet)
  2. Re-splits every chunk["text"] into ~250-word windows (40-word overlap)
  3. Carries every metadata field forward (citation, high_doctrine, year, etc.)
  4. Assigns fresh, unique chunk_ids:  <original_chunk_id>_<NN>
  5. Writes the new manifest.json (overwrites; backup is safe)

After running this, re-run furton_embed.py to regenerate the vectors.

Usage:
    python furton_rechunk.py
"""

import json
import shutil
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't print ✓/✗ — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_WORDS  = 250   # words per sub-chunk (fits in the 256-token model window)
OVERLAP_WORDS = 40    # words shared between adjacent sub-chunks

LIBRARIES = {
    "Warren Buffett":        Path.home() / "Downloads" / "buffett_library",
    "Leopold Aschenbrenner": Path.home() / "aschenbrenner_library",
    "Howard Marks":          Path.home() / "marks_library",
    "Joel Greenblatt":       Path.home() / "greenblatt_library",
    "Cathie Wood":           Path.home() / "wood_library",
}


def split_words(words, target, overlap):
    """Sliding-window split. Returns a list of word-lists, each <= target long,
    with `overlap` words shared between neighbors. The final window covers the tail."""
    if len(words) <= target:
        return [words]
    step = target - overlap
    out = []
    i = 0
    n = len(words)
    while i < n:
        out.append(words[i:i + target])
        if i + target >= n:
            break
        i += step
    return out


def rechunk_manifest(investor_name, lib_path):
    manifest_path = lib_path / "manifest.json"
    backup_path   = lib_path / "manifest_backup.json"

    if not manifest_path.exists():
        print(f"  ⚠ {investor_name}: manifest not found at {manifest_path}, skipping")
        return False

    # 1) Back up once — never clobber an existing good backup
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
        print(f"  ✓ {investor_name}: backed up -> manifest_backup.json")
    else:
        print(f"  • {investor_name}: backup already exists (left untouched)")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    old_chunks = manifest["chunks"]
    new_chunks = []

    for parent in old_chunks:
        words = parent["text"].split()
        windows = split_words(words, TARGET_WORDS, OVERLAP_WORDS)
        sub_total = len(windows)
        for j, w in enumerate(windows):
            sub = dict(parent)                      # carry ALL metadata forward
            sub["chunk_id"]   = f"{parent['chunk_id']}_{j:02d}"
            sub["text"]       = " ".join(w)
            sub["word_count"] = len(w)
            sub["parent_chunk_id"] = parent["chunk_id"]
            sub["sub_index"]  = j
            sub["sub_total"]  = sub_total
            new_chunks.append(sub)

    # Update top-level manifest metadata to reflect the new chunking
    manifest["chunks"]            = new_chunks
    manifest["total_chunks"]      = len(new_chunks)
    manifest["total_words"]       = sum(c["word_count"] for c in new_chunks)
    manifest["chunk_size_target"] = TARGET_WORDS
    manifest["chunk_overlap"]     = OVERLAP_WORDS

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    wc = [c["word_count"] for c in new_chunks]
    print(f"    {len(old_chunks)} chunks -> {len(new_chunks)} sub-chunks "
          f"(words/chunk min={min(wc)} mean={sum(wc)//len(wc)} max={max(wc)})")
    return True


def main():
    print("=" * 60)
    print("FURTON RESEARCH — Re-Chunker (250 words / 40 overlap)")
    print("=" * 60)
    ok = 0
    for investor_name, lib_path in LIBRARIES.items():
        if rechunk_manifest(investor_name, lib_path):
            ok += 1
    print("\n" + "=" * 60)
    print(f"DONE — {ok}/{len(LIBRARIES)} libraries re-chunked")
    print("=" * 60)
    print("\nNext: run  python furton_embed.py  to regenerate the vectors,")
    print("then restart furton_server.py (it should print 5/5 using RAG).")


if __name__ == "__main__":
    main()
