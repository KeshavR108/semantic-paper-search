"""Compare this HNSW implementation against hnswlib with identical data and parameters.

Both indexes are built single-threaded on the same vectors, then queried at several
`ef` values. Recall is measured against exact brute-force results. Output is a JSON
file in results/ that also records the hardware, library versions and git commit.

Example:
    python bench/compare_hnswlib.py --metric l2 --n 10000 --dim 64
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import hnswlib
import numpy as np

from env_info import environment
from hnsw import HNSWIndex, knn


def recall_at_k(found, truth):
    k = truth.shape[1]
    return float(np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)]))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    p.add_argument("--n", type=int, default=10_000, help="number of base vectors")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    Q = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
    truth, _ = knn(Q, X, args.k, args.metric)

    print(f"building mine ({args.n} x {args.dim}, {args.metric}) ...", flush=True)
    mine = HNSWIndex(args.dim, metric=args.metric, M=args.M,
                     ef_construction=args.ef_construction, seed=args.seed)
    _, mine_build = timed(lambda: mine.add_batch(X))

    print("building hnswlib ...", flush=True)
    ref = hnswlib.Index(space=args.metric, dim=args.dim)
    ref.init_index(max_elements=args.n, M=args.M, ef_construction=args.ef_construction,
                   random_seed=args.seed)
    _, ref_build = timed(lambda: ref.add_items(X, num_threads=1))

    rows = []
    for ef in args.ef:
        if ef < args.k:
            continue
        (m_ids, _), m_t = timed(lambda: mine.batch_search(Q, args.k, ef))
        ref.set_ef(ef)
        (r_ids, _), r_t = timed(lambda: ref.knn_query(Q, k=args.k, num_threads=1))
        row = {
            "ef": ef,
            "mine_recall": recall_at_k(m_ids, truth),
            "hnswlib_recall": recall_at_k(r_ids, truth),
            "mine_qps": args.queries / m_t,
            "hnswlib_qps": args.queries / r_t,
        }
        rows.append(row)
        print(f"ef={ef:4d}  recall mine {row['mine_recall']:.4f}  hnswlib {row['hnswlib_recall']:.4f}"
              f"  |  QPS mine {row['mine_qps']:,.0f}  hnswlib {row['hnswlib_qps']:,.0f}", flush=True)

    result = {
        "benchmark": "compare_hnswlib",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "random_gaussian", "n": args.n, "dim": args.dim,
                    "queries": args.queries, "seed": args.seed},
        "params": {"metric": args.metric, "k": args.k, "M": args.M,
                   "ef_construction": args.ef_construction, "threads": 1},
        "build_seconds": {"mine": mine_build, "hnswlib": ref_build},
        "by_ef": rows,
        "max_recall_gap_points": max(100 * (r["hnswlib_recall"] - r["mine_recall"]) for r in rows),
        "environment": environment(extra_packages=["hnswlib"]),
    }

    out = Path(args.out_dir) / f"compare_hnswlib_{args.metric}_n{args.n}_d{args.dim}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
