#!/usr/bin/env python3
"""
Furton Research — Embedding Generator (RAG Pipeline, Step 1)
=============================================================
Reads all five investor libraries, embeds every chunk with a local
model (all-MiniLM-L6-v2), and writes the vectors to disk alongside
each manifest.

Run this ONCE after building/updating any library. It runs fully
offline after the first download of the model (~90 MB).

Output per library:
    <library>/embeddings.npy    — float32 matrix (n_chunks × 384)
    <library>/embeddings_meta.json — chunk_id order + model info

Usage:
    python furton_embed.py

Requirements:
    pip install sentence-transformers numpy
    (this pulls in PyTorch — a ~2 GB one-time install)
"""

import json
import time
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"   # 22M params, 384-dim, fast on CPU
EMBED_DIM  = 384

LIBRARIES = {
    "Warren Buffett":        Path.home() / "Downloads" / "buffett_library",
    "Leopold Aschenbrenner": Path.home() / "aschenbrenner_library",
    "Howard Marks":          Path.home() / "marks_library",
    "Joel Greenblatt":       Path.home() / "greenblatt_library",
    "Cathie Wood":           Path.home() / "wood_library",
}


def load_model():
    """Load the embedding model. Downloads ~90 MB on first run, then offline."""
    print(f"Loading embedding model '{MODEL_NAME}'...")
    print("  (first run downloads ~90 MB; subsequent runs are offline)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model loaded ({EMBED_DIM}-dim vectors)\n")
    return model


def embed_library(investor_name, lib_path, model):
    manifest_path = lib_path / "manifest.json"
    if not manifest_path.exists():
        print(f"  ⚠ {investor_name}: manifest not found at {manifest_path}, skipping")
        return False

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest["chunks"]
    if not chunks:
        print(f"  ⚠ {investor_name}: no chunks in manifest, skipping")
        return False

    print(f"  {investor_name}: embedding {len(chunks)} chunks...")
    t0 = time.time()

    # Embed the raw text of each chunk
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity becomes a dot product
    ).astype(np.float32)

    # Save vectors
    np.save(lib_path / "embeddings.npy", vectors)

    # Save the chunk order so the server can map rows back to chunks
    meta = {
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "count": len(chunks),
        "normalized": True,
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }
    with open(lib_path / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    print(f"    ✓ {len(chunks)} vectors saved ({dt:.1f}s, {len(chunks)/dt:.0f} chunks/sec)")
    return True


def main():
    print("=" * 60)
    print("FURTON RESEARCH — Embedding Generator (RAG Step 1)")
    print("=" * 60)

    model = load_model()

    print("Embedding libraries...")
    ok = 0
    for investor_name, lib_path in LIBRARIES.items():
        if embed_library(investor_name, lib_path, model):
            ok += 1

    print("\n" + "=" * 60)
    print(f"DONE — {ok}/{len(LIBRARIES)} libraries embedded")
    print("=" * 60)
    print("\nNext: restart furton_server.py — it will detect the embeddings")
    print("and switch to RAG retrieval automatically.")
    print("\nRe-run this script any time you rebuild or update a library.")


if __name__ == "__main__":
    main()
