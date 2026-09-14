# 03 · Embeddings & Semantic Search

Keyword search fails the moment a query and a document describe the same
thing in different words ("car" vs. "automobile," "cheap flights" vs.
"budget airfare"). **Embeddings** — dense vectors that place
semantically-similar text near each other in vector space — fix that.
This module covers getting embeddings, comparing them, storing them at
scale, and when a full RAG pipeline is (and isn't) the right tool.

## Getting embeddings

An embedding model maps a string to a fixed-length vector, independent of
the string's length:

```python
from dotenv import load_dotenv
import voyageai   # a common embeddings provider; any embedding API follows this shape
import numpy as np

load_dotenv()
vo = voyageai.Client()

def embed_texts(texts: list[str], input_type: str = "document") -> np.ndarray:
    result = vo.embed(texts, model="voyage-3", input_type=input_type)
    return np.array(result.embeddings)

docs = [
    "The return policy allows refunds within 30 days of purchase.",
    "Our premium plan includes priority customer support.",
    "You can reset your password from the account settings page.",
]
doc_vectors = embed_texts(docs, input_type="document")
print(doc_vectors.shape)   # (3, 1024) — 3 documents, 1024-dim vectors
```

Note `input_type` — many embedding models distinguish "document" (things
you're indexing) from "query" (things you're searching with) and produce
slightly different vectors optimized for each role; mixing them up
degrades search quality even though both calls succeed without error.

## Cosine similarity

Semantic closeness is measured as the angle between two vectors, not their
raw distance — two vectors pointing the same direction are "similar" even
if one is much longer, so cosine similarity divides out magnitude:

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query_vector = embed_texts(["How do I get my money back?"], input_type="query")[0]

scores = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vectors]
best = docs[int(np.argmax(scores))]
print(best)   # "The return policy allows refunds within 30 days of purchase."
print(scores)
```

Notice the query shares almost no words with the matching document —
"money back" and "refund," "policy" never even appears in the query. This
is the entire value proposition of embeddings over keyword matching.

## Vector stores at scale

Brute-force cosine similarity against every document (as above) is fine
for hundreds of documents; it doesn't scale to millions. A vector store
indexes embeddings for fast approximate nearest-neighbor search:

```python
import chromadb

client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_or_create_collection("support_docs")

collection.add(
    ids=[f"doc_{i}" for i in range(len(docs))],
    embeddings=doc_vectors.tolist(),
    documents=docs,
)

results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=2,
)
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"{distance:.4f}  {doc}")
```

At real scale (millions of vectors), the store uses an approximate index
(HNSW is common) rather than scanning every vector — trading a small,
tunable amount of recall for search times that stay fast as the collection
grows, instead of degrading linearly with size the way brute force does.

## Chunking documents before embedding

A whole 50-page manual embedded as one vector loses too much specificity —
the vector becomes an average over everything in the document, diluting
any single fact. Split documents into chunks small enough to represent one
coherent idea, with a little overlap so a fact split across a chunk
boundary is still findable:

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

manual_chunks = chunk_text(open("product_manual.txt").read())
manual_vectors = embed_texts(manual_chunks)
```

Chunk size is a real tuning knob: too small and a chunk lacks the context
to be individually meaningful; too large and it dilutes back toward the
whole-document-average problem. A few hundred tokens per chunk, with
10-20% overlap, is a reasonable starting point to tune from.

## When to reach for full RAG

Semantic search alone (embed the query, retrieve the top-k chunks, show
them to a user) is often enough. **RAG** (retrieval-augmented generation)
adds a generation step: retrieved chunks are fed into an LLM prompt so it
can synthesize an answer grounded in them, rather than making the user
read the raw chunks themselves:

```python
from anthropic import Anthropic
llm = Anthropic()

def answer_with_rag(question: str, collection, k: int = 3) -> str:
    q_vec = embed_texts([question], input_type="query")[0]
    hits = collection.query(query_embeddings=[q_vec.tolist()], n_results=k)
    context = "\n\n".join(hits["documents"][0])

    resp = llm.messages.create(
        model="claude-sonnet-5", max_tokens=500,
        messages=[{"role": "user", "content":
            f"Answer using ONLY the context below. If the answer isn't in "
            f"the context, say so.\n\nContext:\n{context}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text
```

| Situation | Use |
|---|---|
| User just needs to find the right document/section | Plain semantic search |
| User wants a synthesized answer combining multiple sources | RAG |
| The corpus is small enough to fit in context wholesale | Skip retrieval — just paste it all in (cache it, module 2, Level 2) |
| Answers need to cite exact sources | RAG with retrieved-chunk citations |

## How It Actually Works

An embedding model is a transformer (module 1's architecture, or a
variant) trained with a *different* objective than next-token prediction:
instead of predicting the next token, it's trained (often via contrastive
learning) so that texts humans or automated signals judge as related
produce vectors with high cosine similarity, and unrelated texts produce
vectors with low similarity. Practically, the model still runs tokens
through attention and feed-forward layers exactly as in module 1; the
difference is what happens at the *output* — instead of projecting to
vocabulary-sized logits for next-token prediction, the final hidden states
are pooled (often averaged, or the vector for a designated position) into
one fixed-length vector per input, and training pushes that pooled vector
toward the geometric arrangement the similarity objective wants.

This is why cosine similarity, not Euclidean distance, is the standard
metric: the training objective is typically defined directly in terms of
angle (via a dot-product-based contrastive loss), so the *direction* of
the vector is what's meaningful, not its length — two embeddings of very
different magnitude can still be judged maximally similar if their
directions align.

It's also why chunking matters mechanistically, not just as a heuristic:
pooling collapses all token-level information for the input into one
vector, so anything the pooling step drowns out (a small detail in a
mostly-irrelevant 50-page document) is permanently lost the moment the
vector is computed — there's no way to later "zoom in" on the part of a
long document that mattered, because that information was never separately
preserved past the embedding step. This is the same information-loss logic
as module 7 (Level 2)'s context compaction, applied to embeddings instead
of conversation history.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Embedding | Fixed-length vector; semantically similar text → nearby vectors |
| `input_type` | Query vs. document embeddings are often different — don't mix them up |
| Cosine similarity | Measures angle, not magnitude — matches the training objective |
| Vector store | Approximate nearest-neighbor index for scale beyond brute force |
| Chunking | Balance too-small (loses context) vs. too-large (dilutes meaning) |
| RAG | Retrieval + generation, for synthesized rather than raw-document answers |

## Exercise

Take 10 short paragraphs on different topics, embed them, and build a
brute-force semantic search function using cosine similarity. Verify it
correctly retrieves a paragraph about "canine companions" for the query
"dogs as pets" despite zero shared keywords. Then wrap it in the RAG
pattern above, ask a question the corpus can't answer, and confirm the
model says so rather than fabricating an answer from unrelated chunks.
