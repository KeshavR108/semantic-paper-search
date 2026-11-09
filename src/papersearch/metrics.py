import math


def recall_at_k(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    total = 0.0
    for docs, rel in zip(ranked, relevant):
        total += len(set(docs[:k]) & rel) / len(rel)
    return total / len(ranked)


def mrr_at_k(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    total = 0.0
    for docs, rel in zip(ranked, relevant):
        for rank, doc in enumerate(docs[:k], start=1):
            if doc in rel:
                total += 1.0 / rank
                break
    return total / len(ranked)


def ndcg_at_k(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    total = 0.0
    for docs, rel in zip(ranked, relevant):
        dcg = sum(1.0 / math.log2(rank + 1)
                  for rank, doc in enumerate(docs[:k], start=1) if doc in rel)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
        total += dcg / ideal
    return total / len(ranked)
