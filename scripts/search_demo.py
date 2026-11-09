import argparse
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

import hnsw
from papersearch import BM25, load_scifact
from papersearch.embed import DEFAULT_MODEL, encode, load_model


def load_or_build(data, model, data_dir, M=16, ef_construction=200):
    vec_path = Path(data_dir) / f"scifact_docs_{DEFAULT_MODEL.split('/')[-1]}.npy"
    idx_path = Path(data_dir) / "scifact_index.npz"
    if vec_path.exists():
        vecs = np.load(vec_path)
    else:
        print("encoding papers (first run only) ...", file=sys.stderr)
        vecs = encode(model, data.texts, show_progress=True)
        np.save(vec_path, vecs)
    if idx_path.exists():
        index = hnsw.load(idx_path)
    else:
        print("building HNSW index (first run only) ...", file=sys.stderr)
        index = hnsw.HNSWIndex(vecs.shape[1], metric="cosine", M=M, ef_construction=ef_construction)
        index.add_batch(vecs)
        hnsw.save(index, idx_path)
    return index


def show(label, rows, scores, data, relevant_note):
    print(f"\n  {label}")
    for rank, (row, score) in enumerate(zip(rows, scores), start=1):
        title = textwrap.shorten(data.titles[row], width=88, placeholder="...")
        print(f"   {rank}. [{score:6.3f}] {title}{relevant_note(row)}")


def answer(query, data, bm25, model, index, k, ef):
    t0 = time.perf_counter()
    kw_rows, kw_scores = bm25.search(query, k)
    t_kw = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_vec = encode(model, [query])[0]
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    sem_rows, sem_dists = index.search(q_vec, k, ef)
    t_hnsw = time.perf_counter() - t0

    both = set(kw_rows.tolist()) & set(sem_rows.tolist())
    mark = lambda row: "   <- in both" if row in both else ""
    print(f'\nQuery: "{query}"')
    show(f"Keyword search (BM25), {t_kw * 1e3:.1f} ms", kw_rows, kw_scores, data, mark)
    show(f"Semantic search (MiniLM + HNSW): encode {t_enc * 1e3:.1f} ms + search {t_hnsw * 1e3:.2f} ms",
         sem_rows, 1.0 - sem_dists, data, mark)
    print("\n  Scores: BM25 score for keyword search, cosine similarity for semantic search.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="*", help="search text; omit for interactive mode")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--ef", type=int, default=100)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    data = load_scifact(cache_dir=f"{args.data_dir}/hf")
    bm25 = BM25(data.texts)
    model = load_model()
    index = load_or_build(data, model, args.data_dir)
    ef = max(args.ef, args.k)

    if args.query:
        answer(" ".join(args.query), data, bm25, model, index, args.k, ef)
        return
    print(f"{len(data.texts)} papers indexed. Type a query (empty line to quit).")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            break
        if not query:
            break
        answer(query, data, bm25, model, index, args.k, ef)


if __name__ == "__main__":
    main()
