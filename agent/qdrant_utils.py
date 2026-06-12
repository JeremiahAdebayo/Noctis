from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import os

QDRANT_URL = "https://dc231e54-2819-4711-9de9-2dc2db3f0669.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_client = None
_embedder = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"[Qdrant]: Created collection '{COLLECTION_NAME}'")
    else:
        print(f"[Qdrant]: Collection '{COLLECTION_NAME}' already exists")

def index_chunks(chunks: list[dict]):
    """
    chunks: list of dicts with keys:
        name, type, path, source, start_line, end_line
    """
    client = get_client()
    embedder = get_embedder()

    texts = [
        f"{c['type']} {c['name']} in {c['path']}:\n{c['source']}"
        for c in chunks
    ]
    vectors = embedder.encode(texts, show_progress_bar=False).tolist()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "name": chunks[i]["name"],
                "type": chunks[i]["type"],
                "path": chunks[i]["path"],
                "source": chunks[i]["source"],
                "start_line": chunks[i]["start_line"],
                "end_line": chunks[i]["end_line"],
            }
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Qdrant]: Indexed {len(points)} chunks")

def retrieve_chunks(query: str, top_k: int = 8) -> list[dict]:
    client = get_client()
    embedder = get_embedder()

    vector = embedder.encode([query])[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True
    )

    return [hit.payload for hit in results]

def clear_collection():
    """Call this before indexing a new repo so stale chunks don't bleed in."""
    client = get_client()
    client.delete_collection(COLLECTION_NAME)
    ensure_collection()
    print(f"[Qdrant]: Collection cleared and recreated")