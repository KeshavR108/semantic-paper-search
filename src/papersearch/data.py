from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class RetrievalDataset:
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, set[str]]


def load_scifact(cache_dir="data/hf", split="test") -> RetrievalDataset:
    corpus = load_dataset("BeIR/scifact", "corpus", cache_dir=cache_dir)["corpus"]
    queries = load_dataset("BeIR/scifact", "queries", cache_dir=cache_dir)["queries"]
    judged = load_dataset("BeIR/scifact-qrels", cache_dir=cache_dir)[split]

    qrels: dict[str, set[str]] = {}
    for row in judged:
        if row["score"] > 0:
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    text_of = {str(q["_id"]): q["text"] for q in queries}
    query_ids = sorted(qrels, key=int)

    return RetrievalDataset(
        doc_ids=[str(d["_id"]) for d in corpus],
        titles=[d["title"] for d in corpus],
        texts=[(d["title"] + ". " + d["text"]).strip() for d in corpus],
        query_ids=query_ids,
        queries=[text_of[q] for q in query_ids],
        qrels=qrels,
    )
