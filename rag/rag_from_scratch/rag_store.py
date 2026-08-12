import json
from pathlib import Path
import numpy as np
from rag_utils import build_chunks, project_paths, build_chunks

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ===========================================================================
# 1. Embedding
# ===========================================================================

class Embedder:
    """Thin wrapper over sentence-transformers that always returns unit vectors.

    Normalising to length 1 at encode time is not cosmetic: it means cosine
    similarity == dot product, so both the numpy search and the FAISS inner
    product index compute cosine similarity for free.
    """

    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # renamed in sentence-transformers 5.x; the old name still works but warns
        self.dimension = (self.model.get_embedding_dimension()
                          if hasattr(self.model, "get_embedding_dimension")
                          else self.model.get_sentence_embedding_dimension())

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """Embed a list of texts -> (n, dim) float32 array of unit vectors.

        Batching matters: one call on a list of 44 texts is several times
        faster than 44 calls in a Python loop, for identical output.
        """
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,          # <- the unit-length trick
            show_progress_bar=show_progress_bar,
        )
        # float32 halves the memory of float64 and is what FAISS expects.
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, query):
        """Embed a single query -> (dim,) vector.

        The query MUST go through the same model as the documents. Vectors from
        two different models live in unrelated spaces and comparing them
        produces confident nonsense.
        """
        return self.encode([query])[0]


# ===========================================================================
# 2. Similarity, by hand
# ===========================================================================

def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors, written out in full."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:                        # a zero vector has no direction
        return 0.0
    return float(np.dot(a, b) / denominator)


def cosine_similarity_batch(query_vector, matrix):
    """Cosine similarity of one query against every row of a matrix.

    matrix: (n, dim). Returns (n,). Vectorised, so one numpy call replaces a
    Python loop over n vectors - the difference is roughly 100x.
    """
    query_vector = np.asarray(query_vector, dtype="float32")
    matrix = np.asarray(matrix, dtype="float32")
    query_norm = np.linalg.norm(query_vector)
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    denominator[denominator == 0] = 1e-12       # never divide by zero
    return (matrix @ query_vector) / denominator


def top_k_indices(scores, k):
    """Indices of the k highest scores, best first.

    argpartition is O(n) and only guarantees that the top k end up on the left;
    we sort just those k. Sorting all n would be O(n log n) for no reason.
    """
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    partition = np.argpartition(-scores, k - 1)[:k]   # top k on the left, unordered
    return partition[np.argsort(-scores[partition])]  # then sort just those k


# ===========================================================================
# 3. The vector store
# ===========================================================================

class VectorStore:
    """Chunks + their vectors + search. This is a vector database, minus the
    server, the persistence guarantees and the network protocol.

    The invariant that makes it work: row i of `vectors` is the embedding of
    `chunks[i]`. Break the alignment and your citations point at the wrong
    document while everything still *looks* like it works.
    """

    def __init__(self, chunks=None, vectors=None, model_name=DEFAULT_MODEL):
        self.chunks = list(chunks) if chunks is not None else []
        self.vectors = (np.asarray(vectors, dtype="float32")
                        if vectors is not None else np.zeros((0, 0), dtype="float32"))
        if len(self.chunks) != len(self.vectors):
            raise ValueError(
                f"alignment broken: {len(self.chunks)} chunks vs "
                f"{len(self.vectors)} vectors")
        self.model_name = model_name
        self.index = None                        # FAISS index, built on demand

    # -- construction -------------------------------------------------------

    @classmethod
    def from_chunks(cls, chunks, embedder, batch_size=32, show_progress_bar=True):
        """Embed every chunk and wrap the result in a store."""
        chunks = list(chunks)
        vectors = embedder.encode([c["text"] for c in chunks],
                                  batch_size=batch_size,
                                  show_progress_bar=show_progress_bar)
        return cls(chunks, vectors, model_name=embedder.model_name)

    def add(self, chunks, vectors):
        """Append new chunks. Invalidates the FAISS index (rebuilt on demand)."""
        chunks = list(chunks)
        vectors = np.asarray(vectors, dtype="float32")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if self.vectors.size and vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"dimension mismatch: store is {self.vectors.shape[1]}-d, "
                f"got {vectors.shape[1]}-d. Did you change embedding model?")
        self.chunks.extend(chunks)
        self.vectors = vectors if not self.vectors.size else np.vstack([self.vectors, vectors])
        self.index = None
        return self

    def __len__(self):
        return len(self.chunks)

    @property
    def dimension(self):
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    # -- search -------------------------------------------------------------

    def search_numpy(self, query_vector, top_k=5):
        """Exact brute-force search: score everything, keep the best k."""
        if not len(self):
            return []
        scores = cosine_similarity_batch(query_vector, self.vectors)
        return [self._hit(index, scores[index], rank)
                for rank, index in enumerate(top_k_indices(scores, top_k), start=1)]

    def build_faiss(self):
        """Build a flat inner-product index over the (already unit) vectors."""
        import faiss  # lazy: faiss-cpu is a big import

        if not len(self):
            raise ValueError("cannot build an index over an empty store")
        index = faiss.IndexFlatIP(self.dimension)   # IP = inner product = cosine here
        index.add(self.vectors)                     # float32 and C-contiguous
        self.index = index
        return index

    def search_faiss(self, query_vector, top_k=5):
        """Same maths as search_numpy, run by FAISS. Exact on a flat index."""
        if not len(self):
            return []
        if self.index is None:
            self.build_faiss()
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, len(self)))
        return [self._hit(int(index), float(score), rank)
                for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1)
                if index != -1]

    def search(self, query_vector, top_k=5, min_score=None, backend="auto"):
        """Search with whichever backend is available, then optionally filter.

        min_score drops weak matches. A RAG system that always returns its top 5
        will happily feed the LLM five irrelevant chunks for an off-topic
        question - the threshold is how it learns to say "I don't know".
        Calibrate it on YOUR corpus; there is no universal value.
        """
        if backend == "numpy":
            hits = self.search_numpy(query_vector, top_k)
        elif backend == "faiss":
            hits = self.search_faiss(query_vector, top_k)
        else:
            try:
                hits = self.search_faiss(query_vector, top_k)
            except ImportError:                  # no faiss installed - numpy still works
                hits = self.search_numpy(query_vector, top_k)
        if min_score is not None:
            hits = [hit for hit in hits if hit["score"] >= min_score]
        return hits

    def _hit(self, index, score, rank):
        """One search result: the score, the text to feed the LLM, the metadata
        to cite it with."""
        chunk = self.chunks[index]
        return {"rank": rank, "score": float(score), "index": int(index),
                "id": chunk.get("id"), "text": chunk["text"],
                "metadata": chunk.get("metadata", {})}

    # -- persistence --------------------------------------------------------

    def save(self, directory):
        """Persist to plain files: vectors.npy + chunks.json + store_config.json.

        Deliberately boring formats. You can inspect every one of them with a
        text editor or numpy, which is exactly what you want when a retrieval
        result looks wrong. The FAISS index is not saved: rebuilding a flat
        index is cheap, and a stale index is worse than no index.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "store_config.json").write_text(json.dumps({
            "model_name": self.model_name,
            "dimension": self.dimension,
            "n_chunks": len(self),
        }, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory):
        """Rebuild a store from the three files that save() wrote."""
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        config_path = directory / "store_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        return cls(chunks, vectors, model_name=config.get("model_name", DEFAULT_MODEL))


# ===========================================================================
# 4. Convenience
# ===========================================================================

def build_or_load_store(paths, embedder=None, chunk_size=500, chunk_overlap=50,
                        rebuild=False):
    """Load the saved store if it matches the current model, else build it.

    `paths` is a project_paths() dict. This is the one call Session 3 needs to
    get from "a folder of documents" to "a searchable store".
    """

    storage = Path(paths["storage"])
    if not rebuild and (storage / "vectors.npy").exists():
        store = VectorStore.load(storage)
        if embedder is None or store.model_name == embedder.model_name:
            return store
        print("Saved store was built with a different model - rebuilding.")

    if embedder is None:
        embedder = Embedder()
    chunks = build_chunks(paths["documents"], chunk_size, chunk_overlap)
    store = VectorStore.from_chunks(chunks, embedder)
    store.save(storage)
    return store


def format_hits(hits, width=200):
    """Readable multi-line summary of search results, for printing."""
    lines = []
    for position, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        page = f", page {meta['page']}" if meta.get("page") else ""
        text = " ".join(hit["text"].split())
        text = text if len(text) <= width else text[:width] + " ..."
        lines.append(f"[{hit.get('rank', position)}] score={hit['score']:.3f}  "
                     f"({meta.get('source', '?')}{page})\n    {text}")
    return "\n".join(lines) if lines else "(no results above the score threshold)"


if __name__ == "__main__":
    paths = project_paths()
    embedder = Embedder()
    store = build_or_load_store(paths, embedder, rebuild=True)
    print(f"{len(store)} chunks, {store.dimension} dims -> {paths['storage']}")
    print(format_hits(store.search(embedder.encode_query("what is attention?"), top_k=5, min_score=0.6)))
