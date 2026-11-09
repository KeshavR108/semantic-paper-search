# Semantic Paper Search

A search engine for scientific papers that matches on meaning instead of keywords.
Search for "heart attack risk in overweight people" and it returns "Body-mass index and
mortality in Korean men and women" and "Obesity as a medical problem", papers that share
almost no words with the query. Keyword search on the same query returns a paper about a
heart disease questionnaire, because it matched "heart", "risk" and "people".

I'm building the core pieces myself instead of calling a library for them: the vector
index that makes search fast, the keyword baseline it's compared against, and the
evaluation. The index and a first working version of the search are done. Fine-tuning the
embedding model is what I'm working on now.

## How it works

The idea behind semantic search is to turn every piece of text into a vector, a list of a
few hundred numbers, in a way where texts with similar meanings end up close together.
Once you have that, searching is a geometry problem: turn the query into a vector and find
the stored vectors nearest to it.

That splits the project into three parts.

**1. The embedding model** turns text into vectors. I'm starting from MiniLM, a small
transformer, and fine-tuning it on arXiv papers, using each paper's title as a query and
its abstract as the matching document. The training objective is contrastive: pull each
title's vector toward its own abstract and away from every other abstract in the batch.
*(In progress.)*

**2. The vector index** finds the nearest vectors quickly. Comparing a query against every
stored vector works, but it's linear in the size of the collection and gets slow fast. I
implemented HNSW (Hierarchical Navigable Small World), the graph-based algorithm used inside
most vector databases. *(Done, in Python. C++ port in progress.)*

**3. Evaluation.** For the index, I check recall: of the true nearest neighbors, how many
did HNSW actually find, and how fast. For the full search system, I use SciFact, a
benchmark of 5,183 paper abstracts where people have marked which ones are relevant to each
of 300 scientific claims. The baselines are BM25 keyword search (which I also wrote from
scratch) and the original MiniLM; the fine-tuned model will be compared against both.

## The HNSW index

HNSW builds a graph over the vectors in several layers. The bottom layer contains every
vector, each linked to a handful of its nearest neighbors. Each layer above contains a random,
much smaller subset of the one below, so the top layers have only a few nodes with long links
between them. It works a lot like a road network, with highways on top and local streets at
the bottom.

A search starts at the top layer and greedily moves to whichever neighbor is closest to the
query, until it can't get any closer. Then it drops down a layer and repeats. On the bottom
layer it switches to a wider beam search that keeps the best `ef` candidates around instead
of just one, which is what lets it get past dead ends. The upper layers get you to the right
neighborhood in a few jumps, and the bottom layer does the careful work.

When a new vector is inserted, it runs that same search to find its neighbors and links to
them in both directions. One detail turned out to matter a lot: which neighbors a node keeps.
Keeping only the closest ones sounds right, but it makes nodes in dense areas link only to
each other and cuts off everything else. The paper's heuristic only keeps a neighbor if it
isn't already reachable through a neighbor that was kept, so links end up pointing in
different directions. In my tests on 5,000 vectors, the simple rule left 21 nodes
unreachable from the entry point. With the heuristic, every node was reachable.

The main parameters:

| Parameter | What it controls | Raising it |
|---|---|---|
| `M` | links per node (twice that on the bottom layer) | better recall, more memory |
| `ef_construction` | beam width while building | better graph, slower build |
| `ef` | beam width while searching (must be ≥ k) | better recall, slower queries |

## Results so far

All numbers come from scripts in `bench/`, which write to `results/` along with the
hardware, library versions, and commit they were run on.

### Search quality on SciFact

5,183 abstracts, 300 test queries, pretrained `all-MiniLM-L6-v2` (not fine-tuned yet),
HNSW with M=16 and ef_construction=200, single thread, Apple M4:

| System | nDCG@10 | MRR@10 | Recall@100 | Search time (p50) |
|---|---|---|---|---|
| BM25 keyword search | 0.664 | 0.632 | 0.886 | 0.31 ms |
| MiniLM + exact search | 0.648 | 0.607 | 0.925 | 5.02 ms |
| MiniLM + HNSW (ef=10) | 0.630 | 0.591 | n/a | 0.14 ms |
| MiniLM + HNSW (ef=20) | 0.644 | 0.604 | n/a | 0.21 ms |
| MiniLM + HNSW (ef=50) | 0.648 | 0.607 | n/a | 0.41 ms |
| MiniLM + HNSW (ef=100) | 0.648 | 0.607 | 0.928 | 0.69 ms |

Three things I took from this:

- **Keyword search is a strong baseline.** BM25 beats the pretrained embedding model on
  nDCG@10 (0.664 vs 0.648). SciFact queries are full of specific terms like gene and drug
  names, which is where exact word matching does well. The embedding model finds more of
  the relevant papers overall (recall@100 of 0.925 vs 0.886) but ranks them a little worse.
  This is the gap fine-tuning is supposed to close.
- **Approximate search costs nothing here once ef is 50 or more.** The HNSW rows use the same
  embeddings as the exact row, so any difference is HNSW missing true neighbors. At ef=10 it
  finds 92% of the true top 10 and nDCG drops by 0.018; at ef=50 it finds 99% and the
  quality scores are identical to exact search.
- **The index isn't the slow part.** Turning a query into a vector takes about 5 ms on CPU;
  the HNSW lookup takes well under 1 ms. At this size (about 5,000 documents) exact search
  would also be fast enough, so the index only starts to matter on larger collections.

Recall@100 is n/a where ef is below 100, since the search can't return more results than ef.

### Index recall against hnswlib

Recall against hnswlib, the standard C++ implementation, with the same data and settings (10,000 random 64-dimensional vectors, 200 held-out queries, M=16, ef_construction=200, single thread, Apple M4):

| Metric | ef | Recall@10 (mine) | Recall@10 (hnswlib) | QPS (mine) | QPS (hnswlib) |
|---|---|---|---|---|---|
| l2 | 10 | 0.434 | 0.417 | 7,488 | 100,840 |
| l2 | 50 | 0.850 | 0.843 | 2,532 | 31,638 |
| l2 | 100 | 0.946 | 0.939 | 1,393 | 17,579 |
| l2 | 200 | 0.988 | 0.983 | 772 | 9,749 |
| cosine | 10 | 0.362 | 0.347 | 7,189 | 111,421 |
| cosine | 50 | 0.766 | 0.746 | 2,436 | 31,185 |
| cosine | 100 | 0.906 | 0.908 | 1,396 | 16,800 |
| cosine | 200 | 0.983 | 0.984 | 781 | 9,375 |

Recall matches the reference closely at every setting. Speed doesn't yet: hnswlib is compiled C++ and answers queries 12 to 15 times faster than my Python version. Closing that gap is what the C++ port is for.

The test data here is random Gaussian vectors, which is close to a worst case for any
nearest-neighbor index: in 64 dimensions, random points are all roughly the same distance
apart, so there's little structure for the graph to use. The point of this comparison is
correctness, i.e. whether my implementation behaves the same as the reference library. Benchmarks on
real data (SIFT and GloVe vectors) are next.

## Status

- [x] HNSW in Python and NumPy: insert, search, neighbor heuristic, save and load
- [x] Test suite: graph invariants, exact match on small data, property-based tests, recall vs. hnswlib
- [x] BM25 keyword search from scratch
- [x] Semantic search on SciFact: BM25 and pretrained MiniLM baselines, exact vs. HNSW
- [x] Retrieval metrics from scratch (nDCG, MRR, recall) and a command-line search demo
- [ ] Fine-tuned embedding model, evaluated against both baselines
- [ ] Benchmarks on SIFT-128 and GloVe-100 (recall vs. queries per second)
- [ ] C++ version of the index with Python bindings (pybind11)

## Running it

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Using the index:

```python
import numpy as np
from hnsw import HNSWIndex

vectors = np.random.default_rng(0).standard_normal((10_000, 64)).astype(np.float32)

index = HNSWIndex(dim=64, metric="cosine", M=16, ef_construction=200)
index.add_batch(vectors)

ids, distances = index.search(vectors[0], k=10, ef=100)
```

Searching the papers (installs PyTorch and downloads SciFact, about 8 MB, on first run):

```bash
.venv/bin/pip install -e ".[dev,search]"
.venv/bin/python scripts/search_demo.py "heart attack risk in overweight people"
```

Reproducing the results:

```bash
.venv/bin/python bench/eval_scifact.py
.venv/bin/python bench/compare_hnswlib.py --metric l2
.venv/bin/python bench/compare_hnswlib.py --metric cosine
```

## Layout

```
src/hnsw/
  distance.py      L2 and cosine distance (single, one-to-many, all-pairs)
  brute_force.py   exact nearest neighbors, used as ground truth
  index.py         the HNSW index
  io.py            save and load an index
src/papersearch/
  bm25.py          BM25 keyword search
  metrics.py       nDCG, MRR, recall
  embed.py         text to vectors (sentence-transformers)
  data.py          SciFact loader
scripts/           command-line search demo
tests/             unit tests, graph invariants, recall, save/load, BM25, metrics
bench/             benchmark scripts
results/           benchmark output (JSON)
```

The index itself depends only on NumPy. hnswlib is used in tests and benchmarks as a
reference, never in the implementation. PyTorch and sentence-transformers are only needed
for the embedding model.

## Reference

Yu. A. Malkov and D. A. Yashunin, *Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs*, 2016. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
