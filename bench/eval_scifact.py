import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from env_info import environment
from hnsw import HNSWIndex, knn
from papersearch import BM25, load_scifact, mrr_at_k, ndcg_at_k, recall_at_k
from papersearch.embed import DEFAULT_MODEL, encode, load_model


def quality(ranked_rows, data, relevant):
    ranked = [[data.doc_ids[i] for i in row] for row in ranked_rows]
    return {
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
        "mrr@10": mrr_at_k(ranked, relevant, 10),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "recall@100": recall_at_k(ranked, relevant, 100) if len(ranked[0]) >= 100 else None,
    }


def per_query_ms(fn, items):
    results, times = [], []
    for item in items:
        t0 = time.perf_counter_ns()
        results.append(fn(item))
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return results, times


def latency(times):
    return {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99)),
            "qps": 1000.0 / float(np.mean(times))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    data = load_scifact(cache_dir=f"{args.data_dir}/hf")
    relevant = [data.qrels[q] for q in data.query_ids]
    n_docs, n_queries = len(data.texts), len(data.queries)
    print(f"SciFact: {n_docs} documents, {n_queries} test queries", flush=True)

    t0 = time.perf_counter()
    bm25 = BM25(data.texts)
    bm25_build = time.perf_counter() - t0
    bm25.search(data.queries[0], 100)
    hits, bm25_times = per_query_ms(lambda q: bm25.search(q, 100)[0], data.queries)
    systems = [{"system": "BM25", **quality(hits, data, relevant), "search": latency(bm25_times)}]

    model = load_model(args.model, args.device)
    cache = Path(args.data_dir) / f"scifact_docs_{args.model.split('/')[-1]}.npy"
    if cache.exists():
        doc_vecs, encode_docs_s = np.load(cache), None
        print(f"loaded cached document embeddings from {cache}", flush=True)
    else:
        print("encoding documents ...", flush=True)
        t0 = time.perf_counter()
        doc_vecs = encode(model, data.texts, show_progress=True)
        encode_docs_s = time.perf_counter() - t0
        np.save(cache, doc_vecs)

    encode(model, data.queries[:8])
    q_list, encode_times = per_query_ms(lambda q: encode(model, [q])[0], data.queries)
    q_vecs = np.stack(q_list)

    knn(q_vecs[:1], doc_vecs, 100, "cosine")
    exact_rows, exact_times = per_query_ms(
        lambda q: knn(q[None, :], doc_vecs, 100, "cosine")[0][0], q_vecs)
    exact_top10 = [row[:10] for row in exact_rows]
    systems.append({"system": "MiniLM + exact search", **quality(exact_rows, data, relevant),
                    "search": latency(exact_times)})

    print("building HNSW index ...", flush=True)
    index = HNSWIndex(doc_vecs.shape[1], metric="cosine", M=args.M,
                      ef_construction=args.ef_construction, seed=args.seed)
    t0 = time.perf_counter()
    index.add_batch(doc_vecs)
    hnsw_build = time.perf_counter() - t0

    for ef in args.ef:
        k = min(100, ef)
        index.search(q_vecs[0], k, ef)
        rows, times = per_query_ms(lambda q: index.search(q, k, ef)[0], q_vecs)
        ann_recall = float(np.mean([len(set(a[:10].tolist()) & set(b.tolist())) / 10
                                    for a, b in zip(rows, exact_top10)]))
        systems.append({"system": f"MiniLM + HNSW (ef={ef})", "ef": ef,
                        **quality(rows, data, relevant),
                        "ann_recall@10_vs_exact": ann_recall, "search": latency(times)})

    for s in systems:
        r100 = f"{s['recall@100']:.4f}" if s["recall@100"] is not None else "   -  "
        print(f"{s['system']:26s} nDCG@10 {s['ndcg@10']:.4f}  MRR@10 {s['mrr@10']:.4f}  "
              f"R@100 {r100}  search p50 {s['search']['p50_ms']:.2f} ms", flush=True)

    result = {
        "benchmark": "eval_scifact",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "BeIR/scifact", "split": "test", "documents": n_docs, "queries": n_queries},
        "params": {"model": args.model, "device": args.device, "dim": int(doc_vecs.shape[1]),
                   "bm25": {"k1": bm25.k1, "b": bm25.b}, "M": args.M,
                   "ef_construction": args.ef_construction, "seed": args.seed, "threads": 1},
        "build_seconds": {"bm25": bm25_build, "hnsw": hnsw_build, "encode_documents": encode_docs_s},
        "encode_query": latency(encode_times),
        "systems": systems,
        "environment": environment(extra_packages=["torch", "sentence-transformers", "datasets"]),
    }
    out = Path(args.out_dir) / "scifact_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
