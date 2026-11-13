# WikiBrain

A localized knowledge engine that answers questions **only** from three company policy
documents. Ask a question, get an answer grounded in the actual policy text, with a citation
showing which document it came from — and an honest "I don't know" when the policies don't cover
it.

```
┌───────────────┐   POST /chat    ┌────────────────────────────────────────────┐
│  Next.js UI   │ ──────────────▶ │                 FastAPI                    │
│   Chat.tsx    │ ◀────────────── │                                            │
└───────────────┘ {answer,        │   retriever ──▶ Embedder ──▶ ChromaDB      │
                   sources}       │       │         (MiniLM)     (cosine)      │
                                  │       ▼                                    │
                                  │   rag.build_context ──▶ Gemini             │
                                  └────────────────────────────────────────────┘
```

## How the RAG loop works here

1. **Ingest** (`server/ingest.py`) splits each policy markdown on `##`/`###` headings, keeping
   the heading attached to its body. Only sections longer than `CHUNK_SIZE` get split further by
   a recursive character splitter. Each chunk carries `{source, heading, chunk_index}`.
2. **Embed** (`server/embedding/`) turns chunks into vectors. Default is local
   `all-MiniLM-L6-v2` (384-dim, CPU, no API key).
3. **Store** (`server/vectorstore.py`) upserts into ChromaDB with cosine distance and
   deterministic ids, so re-ingesting replaces rather than duplicates.
4. **Retrieve** (`server/retriever.py`) embeds the question and pulls the top `TOP_K` chunks,
   converting Chroma's distance into a similarity score.
5. **Generate** (`server/rag.py`) assembles the retrieved chunks into a numbered context block,
   wraps it in a system prompt that forbids outside knowledge and mandates citation, and sends it
   to Gemini.

### Why retrieval rather than pasting all three documents into the prompt

With three documents this corpus fits in a modern context window, and stuffing it would work
today. Retrieval is what makes the system survive document #50: cost and latency stay flat as the
corpus grows, and the model sees only relevant evidence instead of 100k tokens it has to search
through. The honest summary is that RAG is not required at this size — it is required at the size
this is designed to grow into.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, skips ~3GB of CUDA
.venv/bin/pip install -r server/requirements.txt

cp .env.example .env      # then add your GEMINI_API_KEY
```

Get a Gemini API key at <https://aistudio.google.com/apikey>.

### Run locally

```bash
# backend — ingests automatically on first start
.venv/bin/uvicorn server.main:app --reload --port 8000

# frontend, in a second terminal
cd client && npm install && npm run dev
```

Open <http://localhost:3000>.

### Run with Docker

```bash
docker compose up --build
```

The first backend build takes several minutes: it installs torch and bakes the embedding model
into the image so containers start without network access. The vector store persists in the
`chroma` named volume across restarts.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, active collection, chunk count, embedder |
| POST | `/chat` | `{question}` → `{answer, sources}` |
| POST | `/ingest?rebuild=true` | Re-ingest the knowledge base |
| GET | `/debug/retrieve?q=…&k=…` | Retrieval only, no generation |

`/debug/retrieve` is the first place to look when an answer is wrong. If the right chunk isn't
being retrieved, the prompt and the model are not the problem.

## Configuration

All settings live in `server/config.py` and are read from `.env`. The ones worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for `/chat`. Not needed for ingest or retrieval. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `EMBEDDER` | `local` | `local` or `gemini` |
| `TOP_K` | `4` | Chunks fed to the model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Characters |
| `DEBUG_PROMPT` | `false` | Logs the fully assembled prompt on every `/chat` |

### Swapping the embedder

Set `EMBEDDER=gemini` and restart. Because the Chroma collection name is scoped by embedder
(`policies-minilm` vs `policies-gemini`), the new embedder lands in its own empty collection and
re-ingests automatically. This is deliberate: MiniLM's 384-dim vectors and Gemini's 768-dim
vectors are different spaces, and querying one with the other returns nonsense rather than an
error. The collection scoping makes that mistake structurally impossible.

Switching to `gemini` also lets you drop `torch` and `sentence-transformers` from
`requirements.txt`, shrinking the backend image from roughly 2GB to about 200MB.

## Measuring retrieval quality

```bash
.venv/bin/python -m server.eval_retrieval --k 4
```

23 questions, each labelled with the document that should answer it. Reports `top-1` and `hit@k`.
Run it after any change to chunk size, overlap, or `TOP_K` — otherwise tuning is guesswork.

Current baseline: **top-1 23/23, hit@4 23/23** at `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`, across 94
chunks.

This measures retrieval only, and retrieval is the half that can be checked without an API key.
The other half — whether the model stays inside the retrieved text — needs a manual faithfulness
check:

```bash
# should answer with a figure and a citation
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"How many days of annual leave do I get?"}'

# should return "I don't know based on the policies I have." with NO sources
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"What is the dress code?"}'
```

Pick an out-of-scope probe that is genuinely absent from all three documents. A question like
"parental leave in Japan" is a poor test: the leave policy *does* cover parental leave, so
retrieval returns real chunks and a partial answer is correct behaviour rather than a
hallucination.

## Knowledge base

Three fictional policy documents for "Aster Dynamics" live in `server/knowledge/`:
`remote-work-policy.md`, `pto-and-leave-policy.md`, `expense-and-travel-policy.md`. To use your
own, drop markdown files in that directory with `##`/`###` headings and re-run ingest with
`--rebuild`.

```bash
.venv/bin/python -m server.ingest --rebuild
```

## Project layout

```
server/
  config.py            all settings, read from .env
  main.py              FastAPI routes
  ingest.py            markdown -> chunks -> Chroma
  retriever.py         question -> top-k hits
  rag.py               context assembly + grounded prompt
  eval_retrieval.py    hit@k harness
  vectorstore.py       ChromaDB helpers
  embedding/           Embedder interface + local / gemini adapters
  llm/                 LLMProvider interface + gemini adapter
  knowledge/           the policy documents
client/
  app/components/Chat.tsx    the whole UI
```

`rag.py` never imports an SDK, and nothing outside `embedding/` imports a model library. That
indirection is what keeps both backends swappable.

## Related documents

- `daybyday_plan.md` — the original 4-week learning schedule (partly superseded).
- `rag_from_scratch_exercises.md` — hand-build-it-first exercises, kept as a reference appendix.
