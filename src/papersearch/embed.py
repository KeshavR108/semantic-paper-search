import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(name: str = DEFAULT_MODEL, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


def encode(model, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=show_progress)
    return np.asarray(vecs, dtype=np.float32)
