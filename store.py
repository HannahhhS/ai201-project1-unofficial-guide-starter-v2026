"""
Stages 3 and 4 of the pipeline: embedding chunks and retrieving them.

Four things in here are worth knowing about, because they'd quietly break the
rest of the project if they were wrong:

1. The Chroma collection is created with cosine distance, explicitly. Chroma
   defaults to squared L2, and the 0.6 threshold the course uses is calibrated
   against cosine. Getting this wrong makes every distance number meaningless.

2. `search` returns the distance alongside each chunk. Milestone 4 has you
   compare distances, so they have to be visible.

3. The embedding model is the one Chroma bundles, not one loaded through
   `sentence-transformers`. It is the same model — `all-MiniLM-L6-v2`, 384
   dimensions — but it arrives as an ONNX build from Chroma's own CDN, so the
   install needs neither PyTorch nor a reachable Hugging Face. See `_embedder`.

4. `search` is hybrid: semantic and BM25 keyword retrieval, fused by rank. The
   distance on every Result is still a real cosine distance whichever
   retriever found the chunk, which is what lets gate.py stay as it is. See
   `_cosine_distance` for the part that keeps that true.
"""

import math
import os
import re
import shutil
from dataclasses import dataclass

# Must be set BEFORE chromadb is imported. Without it, some Chroma versions
# print "Failed to send telemetry event ..." on every single call — which looks
# exactly like a real error, isn't one, and cost a previous cohort a lot of
# confused help-channel messages.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

import config
from chunker import Chunk


@dataclass
class Result:
    """One retrieved chunk and how far it was from the question."""

    text: str
    source: str
    label: str
    distance: float   # LOWER IS BETTER. 0.3 is close, 0.9 is unrelated.
    produced_by: str


_model = None

# The model Chroma bundles. Anything else in config.EMBEDDING_MODEL means
# "fetch that one from Hugging Face instead" — see `_embedder`.
BUNDLED_MODEL = "all-MiniLM-L6-v2"


class _OnnxEmbedder:
    """
    Chroma's built-in embedder, wrapped to look like the other two.

    Chroma's embedding functions are called directly and hand back numpy
    arrays. The rest of this file wants `.encode(texts)`, so the adapter lives
    here rather than making every caller care which embedder it got.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts, show_progress_bar: bool = False):
        return [vector.tolist() for vector in self._ef(list(texts))]


def _sentence_transformer(name: str):
    """
    The escape hatch: any model that isn't the bundled one.

    Unit 2's "try a second embedding model" stretch option comes through here,
    and so does anything you set `EMBEDDING_MODEL` to. This path *does* need
    `sentence-transformers` and a reachable Hugging Face, neither of which the
    default install has — which is the whole point of the default install.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"config.EMBEDDING_MODEL is set to {name!r}, which isn't the model "
            f"Chroma bundles ({BUNDLED_MODEL!r}), so it has to be downloaded "
            f"from Hugging Face.\n"
            f"Install the optional dependency first:\n"
            f"    pip install 'sentence-transformers>=3.4,<3.5'\n"
            f"Or set EMBEDDING_MODEL back to {BUNDLED_MODEL!r}."
        ) from exc

    return SentenceTransformer(name)


def _embedder():
    """
    Load the embedding model once and keep it.

    First call is slow — it downloads about 80 MB. That's why setup happens
    before class.
    """
    global _model

    if _model is not None:
        return _model

    # Used only by this repo's own smoke test, which runs where no model can be
    # downloaded at all. Never set this yourself.
    if os.getenv("AI201_FAKE_EMBEDDINGS") == "1":
        from _smoke_embedder import FakeEmbedder

        _model = FakeEmbedder()
    elif config.EMBEDDING_MODEL == BUNDLED_MODEL:
        _model = _OnnxEmbedder()
    else:
        _model = _sentence_transformer(config.EMBEDDING_MODEL)

    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Turn text into vectors. Runs on your machine, costs no API quota."""
    vectors = _embedder().encode(texts, show_progress_bar=False)
    # sentence-transformers and the smoke stand-in return something with a
    # .tolist(); _OnnxEmbedder has already done that conversion itself.
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def _client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def build_index(
    chunks: list[Chunk],
    corpus: str | None = None,
    variant: str = "default",
) -> int:
    """
    Embed every chunk and store it.

    `variant` lets you keep more than one index of the same corpus at the same
    time. In unit 2, when you compare two chunking strategies, index the second
    one as variant="v2" and you can query both instead of deleting the first
    and starting over.
    """
    name = config.collection_name(corpus, variant)
    client = _client()

    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        # ⚠️ Do not remove. Chroma defaults to squared L2, and every distance
        # number in this course assumes cosine.
        metadata={"hnsw:space": "cosine"},
    )

    batch = 256
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        collection.add(
            ids=[f"{c.source}#{c.index}" for c in window],
            documents=[c.text for c in window],
            embeddings=embed([c.text for c in window]),
            metadatas=[
                {"source": c.source, "index": c.index, "produced_by": c.produced_by}
                for c in window
            ],
        )

    return len(chunks)


# ─── Keyword retrieval, and fusing it with the semantic kind ─────────────────

_bm25_cache: dict[str, tuple] = {}


def _label(meta) -> str:
    """The id a chunk is known by in both rankings."""
    return f"{meta.get('source', 'unknown')}#{meta.get('index', 0)}"


def _tokenize(text: str) -> list[str]:
    """
    Lowercase runs of letters and digits. `12:30` becomes ['12', '30'].

    Splitting on the colon rather than keeping `12:30` whole is deliberate, and
    measured: a question written "10-15 minutes" has to match a document
    written "10 to 15 minutes", and it only does if both sides break into the
    bare numbers. Keeping punctuation-joined tokens intact ranked the right
    chunk first too, but by a much narrower margin.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_for(collection):
    """
    A BM25 index over the same chunks Chroma already holds.

    BM25 scores a chunk by how many *rare* query terms it contains, so `12:30`
    — in two chunks out of 88 — counts for far more than `dining`, which is in
    a dozen. That is exactly the signal embeddings throw away.

    Built once per collection and kept, because building it means reading every
    document back out of the store. Nothing is re-embedded and nothing is
    written, so `python app.py index` does NOT need re-running to use this.

    ⚠️ The cache is keyed by collection name, so re-indexing inside one
    long-running process would keep serving the stale index. That is serve.py's
    problem only: app.py and run_eval.py exit between indexing and searching.
    """
    if collection.name not in _bm25_cache:
        raw = collection.get(include=["documents", "metadatas", "embeddings"])
        _bm25_cache[collection.name] = (
            BM25Okapi([_tokenize(doc) for doc in raw["documents"]]),
            raw["documents"],
            raw["metadatas"],
            raw["embeddings"],
        )
    return _bm25_cache[collection.name]


def _cosine_distance(a, b) -> float:
    """
    Cosine distance between two vectors, on the same scale Chroma reports.

    This exists because BM25 can surface a chunk the semantic query never
    returned, and such a chunk arrives with no distance attached. gate.py
    compares `min(distance)` against THRESHOLD, so inventing a number here —
    0.0, or 1.0 — would quietly wreck the cutoff measured in Milestone 4.
    Computing the real distance from the stored embedding keeps the gate
    honest; it agrees with Chroma's own numbers to about 1e-9.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int) -> dict[str, float]:
    """
    Combine several ranked lists of labels into one score per label.

    Every list gives each of its labels 1 / (k + rank), and the contributions
    add up. A chunk both retrievers liked therefore outranks one that only a
    single retriever put first, and a chunk neither ranked highly stays down.

    Fusing on RANK rather than on score is the whole point. BM25 scores run
    0-20 on this corpus while cosine distances run 0-1, and they point in
    opposite directions — higher is better for one, lower for the other.
    Normalising them onto a shared scale is possible, but it is fragile and
    needs tuning of its own. Ranks are already comparable.
    """
    scores: dict[str, float] = {}
    for labels in ranked_lists:
        for rank, label in enumerate(labels):
            scores[label] = scores.get(label, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search(
    question: str,
    top_k: int | None = None,
    corpus: str | None = None,
    variant: str = "default",
) -> list[Result]:
    """
    Retrieve the chunks most relevant to a question, nearest-first.

    Two retrievers run over the same chunks and their rankings are fused:
    semantic search, which matches on meaning and is blind to exact tokens, and
    BM25, which matches on rare exact terms and is blind to meaning. They fail
    in opposite directions, which is why using both beats either alone.

    Set config.HYBRID = False for the old semantic-only behaviour; that path is
    unchanged. Either way every Result carries a true cosine distance, so
    gate.py and THRESHOLD are unaffected by which retriever found what.
    """
    top_k = top_k or config.TOP_K
    name = config.collection_name(corpus, variant)

    try:
        collection = _client().get_collection(name)
    except Exception as exc:
        raise RuntimeError(
            f"No index called '{name}'. Run `python app.py index` first."
        ) from exc

    query_vector = embed([question])[0]

    # Hybrid asks for more than top_k so the fusion has something to choose
    # between. Semantic-only still asks for exactly top_k, as it always did.
    wanted = config.HYBRID_CANDIDATES if config.HYBRID else top_k
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=min(wanted, collection.count()),
    )
    semantic = [
        (_label(meta), text, meta, float(distance))
        for text, meta, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    ]

    if not config.HYBRID:
        chosen = semantic[:top_k]
    else:
        # Everything either retriever proposes, by label. Semantic hits arrive
        # with a distance from Chroma; BM25-only hits get one computed below.
        pool = {label: (text, meta, dist) for label, text, meta, dist in semantic}

        bm25, documents, metadatas, embeddings = _bm25_for(collection)
        scores = bm25.get_scores(_tokenize(question))
        by_score = sorted(range(len(documents)), key=lambda i: -scores[i])

        keyword: list[str] = []
        for i in by_score[: min(wanted, len(documents))]:
            if scores[i] <= 0:
                break   # no query term in common at all — not a weak match, no match
            label = _label(metadatas[i])
            keyword.append(label)
            pool.setdefault(
                label,
                (documents[i], metadatas[i],
                 _cosine_distance(query_vector, embeddings[i])),
            )

        fused = _reciprocal_rank_fusion(
            [[label for label, _, _, _ in semantic], keyword], config.RRF_K
        )
        order = sorted(fused, key=lambda label: -fused[label])[:top_k]

        # Fusion decides WHICH top_k chunks come back; they are then handed
        # over nearest-first, like every other caller of this function has
        # always been able to assume (tools/smoke_test.py asserts it, and
        # app.py's retrieve table is read top-down as "closest first"). The
        # recall win is in the membership, not in the order within it.
        chosen = sorted(
            ((label, *pool[label]) for label in order), key=lambda row: row[3]
        )

    results: list[Result] = []
    for label, text, meta, distance in chosen:
        results.append(
            Result(
                text=text,
                source=str(meta.get("source", "unknown")),
                label=label,
                distance=distance,
                produced_by=str(meta.get("produced_by", "unknown")),
            )
        )
    return results


def index_exists(corpus: str | None = None, variant: str = "default") -> bool:
    """Is there an index here to search, without searching it?

    `serve.py`'s health check asks this. It deliberately does not embed
    anything: loading the embedding model takes 80 MB and a few seconds, and a
    health check that heavy is a health check nobody can afford to call.
    """
    try:
        collection = _client().get_collection(config.collection_name(corpus, variant))
        return collection.count() > 0
    except Exception:
        return False


def reset():
    """Delete every index. Occasionally the fastest way out of a mess."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
