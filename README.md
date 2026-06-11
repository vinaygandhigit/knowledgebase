# Knowledge Intelligence (RAG) Platform (FastAPI + ChromaDB + Claude)

Transform Enterprise Documents into an AI-Powered Knowledge Intelligence Platform with ingestion, hybrid search, and LLM response generation

## Features

**Ingestion Layer:**
- Recursive folder scanning using pathlib
- Supported file types: PDF, DOCX, PPTX, XLSX, TXT, MD, YAML, YML, EML, SH, TOML
- Document parsing with dedicated parsers per format
- SHA-256 checksums for change detection
- Semantic chunking with LangChain SemanticChunker, with size guard-rails
  (oversized chunks split with overlap, tiny fragments merged) so chunks fit
  the embedding model window and stay retrieval-friendly
- Tables are isolated and preserved intact through chunking so they re-render
  faithfully; images/diagrams are extracted to disk as retrievable assets
- Embedding generation with configurable provider architecture
- **ChromaDB** local persistent vector store (cosine space) — no external DB server
- Structured logging with correlation ID, job ID, file ID, and execution time

**Retrieval & Generation Layer:**
- Hybrid search: combines vector (semantic) and keyword (BM25) search
- Configurable alpha-weighted Reciprocal Rank Fusion for hybrid blending
- Anthropic Claude (default `claude-opus-4-8`) for grounded response generation,
  with adaptive thinking; Ollama retained as an offline fallback
- Visual-aware retrieval: PDF/DOCX/PPTX/XLSX images and architecture diagrams
  are extracted to disk and tables are preserved, so the Streamlit UI can render
  the images, diagrams, and tables tied to each retrieved chunk
- Chunk-level provenance tracking (file name, score, search type, page/slide/section)
- Input validation and basic path traversal protections

## Project Structure

```text
app/
  api/
    routes.py
    schemas.py
  core/
    config.py
    container.py
    logging.py
    security.py
  domain/
    models.py
  services/
    embeddings/
      base.py
      factory.py
      sentence_transformer_provider.py
    parsers/
      assets.py
      base.py
      factory.py
      pdf_parser.py
      docx_parser.py
      pptx_parser.py
      xlsx_parser.py
      markdown_parser.py
      txt_parser.py
      text_like_parser.py
      eml_parser.py
    chunking.py
    chroma_repository.py
    ingestion_service.py
    llm_provider.py
    retrieval.py
    scanner.py
  main.py
  run_ingestion.py        # console ingestion entry point
eval/
  rag_eval.py             # retrieval + answer-quality evaluator
  sample_eval_dataset.jsonl
ui/
  streamlit_app.py        # chat front-end over the retrieval API
.env
requirements.txt
```

## Setup

1. Create and activate a Python 3.12+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Provide your Anthropic API key (used by the default Claude provider):

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...   # or set ANTHROPIC_API_KEY in .env
   ```

   To run fully offline instead, set `LLM_PROVIDER=ollama` and start Ollama
   (`ollama pull llama3.2:1b && ollama serve`).

4. Configure environment in `.env`:
   - `ROOT_PATH` — folder containing the documents to ingest
   - `CHROMA_PERSIST_DIR` (default `./chroma_store`) and `CHROMA_COLLECTION`
     (default `knowledge_base`)
   - `LLM_PROVIDER` (`claude` default, or `ollama`)
   - `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `CLAUDE_EFFORT` for Claude
   - `EMBEDDING_MODEL_NAME` / `EMBEDDING_DEVICE` for the embedder
   - `HYBRID_SEARCH_ALPHA` (default 0.5 = 50% vector + 50% keyword)

5. Place your documents under `ROOT_PATH`. No database schema or migration is
   required — ChromaDB creates and manages its store automatically.

> **ChromaDB store location.** A relative `CHROMA_PERSIST_DIR` is resolved against
> the project root, so the store lives in one fixed place (`<project>/chroma_store`)
> regardless of the directory you launch from. Use an absolute path to put it
> elsewhere (e.g. a data drive). The API/UI and the ingestion program must point
> at the same store.

## Run

Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Ingest documents — either trigger a reindex via the API (below) or run the
console ingestion program:

```bash
# Full reindex of everything under ROOT_PATH
python -m app.run_ingestion

# Ingest (or re-ingest) a single file without a full reindex
python -m app.run_ingestion --file-path "Knowledge-Base/Queries/q1.docx"

# Override the scan root for this run
python -m app.run_ingestion --root-path "C:/some/other/folder"
```

## Streamlit UI

```bash
pip install streamlit requests
streamlit run ui/streamlit_app.py
```

The UI is a chat front-end over the retrieval API. Under each answer, the
expandable **sources** panel shows every retrieved chunk with its file,
page/slide/section, text, and any attached **images, architecture diagrams, and
tables** — images/diagrams are fetched through `/api/v1/assets`, `[TABLE]`
blocks are rendered as tables, and repeated visuals are de-duplicated per answer.
Set the API Base URL in the sidebar to match the running API.

> **Visuals require a reindex.** Image/diagram/table extraction runs during
> ingestion. After upgrading, trigger a full reindex (`POST /api/v1/ingestion/reindex`
> or the sidebar button) so existing documents are re-parsed with visual
> extraction. Extracted assets are written to a `.kb_assets/` folder beside each
> source document (inside `ROOT_PATH`, so they are servable).
>
> Note: vector image formats (`.wmf`/`.emf`, common in older PowerPoint files)
> are extracted but cannot be displayed inline — the UI offers a download link.

## APIs

### Trigger Full Reindex

- **Method:** POST
- **Endpoint:** `/api/v1/ingestion/reindex`

Response:

```json
{
  "job_id": "8d210e1c-8d87-4f8a-a957-d9f4131f79c2",
  "status": "RUNNING"
}
```

### Ingest a Single File

- **Method:** POST
- **Endpoint:** `/api/v1/ingestion/file`

Request:

```json
{ "file_path": "Knowledge-Base/Queries/q1.docx" }
```

Re-ingests just the given file (chunks/embeddings for that document are replaced)
and refreshes the keyword index, without a full reindex.

### Monitor Job Status

- **Method:** GET
- **Endpoint:** `/api/v1/ingestion/jobs/{job_id}`

### Query Documents (Hybrid Retrieval + LLM Response)

- **Method:** POST
- **Endpoint:** `/api/v1/retrieval/query`

Request:

```json
{
  "query": "What is the main topic covered?",
  "hybrid_alpha": 0.5
}
```

Response:

```json
{
  "query": "What is the main topic covered?",
  "response": "The main topic covered is...",
  "retrieved_chunks": [
    {
      "chunk_id": "abc123",
      "document_id": "doc_001",
      "file_name": "document.pdf",
      "chunk_text": "Page 3 table 1\n[TABLE]\nComponent | Owner\nGateway | Platform",
      "score": 0.95,
      "search_type": "hybrid",
      "source_location": {"page_number": 3, "section_title": "Page 3 table 1"},
      "visual_refs": [
        {"type": "image", "name": "page_3_img_1",
         "path": "/abs/path/Knowledge-Base/queries/page_3_img_1.png",
         "page_number": "3"}
      ]
    }
  ],
  "execution_time_ms": 245.5
}
```

### Serve an Extracted Asset

- **Method:** GET
- **Endpoint:** `/api/v1/assets?path=<absolute path under ROOT_PATH>`

Serves an extracted image/diagram. The path must resolve within `ROOT_PATH`
(path-traversal guarded). Used by the UI to render `visual_refs`.

### Health

- **Method:** GET
- **Endpoint:** `/actuator/health`

## RAG Evaluation Benchmark

Use the evaluator to benchmark **retrieval quality, answer quality, and
performance** against your own question set. It calls the running API per sample.

```bash
# Lexical + retrieval metrics (with a warmup to exclude cold start)
python eval/rag_eval.py --dataset eval/sample_eval_dataset.jsonl --warmup 1

# Add the LLM-as-judge and a concurrent throughput pass
python eval/rag_eval.py --dataset eval/sample_eval_dataset.jsonl \
  --judge --concurrency 4 --repeat 3 --output eval/eval_report.json
```

Dataset formats:
- `.jsonl`: one JSON object per line
- `.json`: top-level array of JSON objects

Supported dataset fields per sample:
- `id` or `sample_id` — unique sample identifier
- `query` — required user query
- `expected_answer` — optional gold answer (enables token-F1, exact match, judge correctness)
- `expected_keywords` — optional keywords expected in the answer / context
- `expected_files` — optional relevant file names (matched by basename)
- `expected_chunk_ids` — optional relevant chunk IDs (chunk-level retrieval metrics)

Metrics reported (averaged over the dataset; `n/a` when the relevant ground
truth is absent):
- **Performance:** throughput (q/s); latency avg / p50 / p95 / p99 / min / max / stdev; API exec avg
- **Retrieval (files & chunks, separately):** Hit@k, Recall@k, Precision@k, MRR (rank-aware)
- **Answer (lexical):** exact match, token-F1, keyword coverage in the **answer** and in the retrieved **context**
- **Answer (LLM-as-judge, `--judge`):** correctness, faithfulness (groundedness), relevance — RAGAS-style 0–1 scores

Key flags: `--warmup` (untimed priming queries), `--repeat` (timed trials per
sample), `--concurrency` (parallel workers for throughput), `--judge` /
`--judge-model` (LLM judge; needs `ANTHROPIC_API_KEY`), `--hybrid-alpha`,
`--timeout-seconds`. Full report is written to `--output` (default
`eval/eval_report.json`).

## Hybrid Search Details

**Vector Search (Semantic):**
- Embeddings from sentence-transformers (default `BAAI/bge-large-en-v1.5`, 1024-dim)
- ChromaDB cosine-distance similarity over INGESTED chunks
- Fast semantic matching

**Keyword Search (BM25):**
- Ranks chunks by relevance to query terms
- Built on startup (and after each reindex) from all ingested chunks
- Complements semantic search for keyword-exact matches

**Alpha Blending (Reciprocal Rank Fusion):**
- `alpha=0.5` (default): 50% semantic + 50% keyword
- `alpha=1.0`: 100% semantic (pure vector search)
- `alpha=0.0`: 100% keyword (pure BM25 search)

## LLM Configuration

The system uses **Anthropic Claude** by default (`claude-opus-4-8`) via the
official `anthropic` SDK:
- Strict grounding: the model answers only from retrieved context
- Adaptive thinking + the `effort` parameter improve grounded counting and
  conflict detection; if the installed SDK or model does not support them, the
  request transparently falls back to a plain call
- Configurable via `.env`: `CLAUDE_MODEL`, `CLAUDE_MAX_TOKENS`,
  `CLAUDE_THINKING`, `CLAUDE_EFFORT`, and `ANTHROPIC_API_KEY`
  (the SDK also resolves `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` from the
  environment if not set in config)

To run on-premise with no external API calls, set `LLM_PROVIDER=ollama` and
configure `OLLAMA_MODEL` / `OLLAMA_BASE_URL`.

Context is built from top-K retrieved chunks — each labelled with its source
file, page/slide, and section — and passed to the LLM for grounded, citable
response generation.

## Metadata JSON

Each chunk stores comprehensive metadata:

- document_id
- file_name
- file_path
- file_type
- checksum
- chunk_id
- chunk_index
- page_number
- slide_number
- section_title
- previous_chunk_id
- next_chunk_id
- source_location
- visual_refs (images/diagrams tied to the chunk)
- created_at
- updated_at

In ChromaDB each chunk is one collection entry: `id` = chunk_id, `document` =
chunk text, `embedding` = vector, and the metadata above is stored alongside
(the full metadata JSON, plus denormalised `document_id` / `file_name` / `status`
used for filtering and provenance).

## Ingestion Performance

Ingestion cost is dominated by **embedding compute**. Tuning levers, highest
impact first:

1. **Use a GPU.** `EMBEDDING_DEVICE=auto` selects CUDA (with FP16) when available;
   `bge-large-en-v1.5` is ~10–50× faster on a GPU than CPU. Force with `cuda`/`cpu`.
2. **Chunking strategy.** `CHUNK_STRATEGY=semantic` (default) embeds every
   sentence to find breakpoints, then embeds the resulting chunks — roughly
   double the embedding work. `CHUNK_STRATEGY=recursive` splits on natural
   boundaries with **no chunk-time embeddings**, cutting embedding work ~in half
   at a modest cost to boundary quality. Switch to `recursive` for large/fast loads.
3. **Batch size.** `BATCH_SIZE` is the embedding encode batch (passed to
   SentenceTransformer). Raise it on a GPU with spare memory (e.g. 64–128).
4. **Worker count.** `WORKER_COUNT` parallelises per-file parse/chunk/store work.
   With a single shared embedding model, keep it low (1–2) on CPU to avoid model
   contention; parsing overlap is the main benefit. A GPU serialises encode
   calls regardless.

A full reindex truncates the collection up front and skips redundant per-file
deletes, so only inserts run during the load.

## Security Controls

- Extension allow-list validation (`SUPPORTED_EXTENSIONS`)
- Max file size validation (`MAX_FILE_SIZE_MB`)
- Path traversal guard — resolved paths (ingestion and asset serving) must remain
  under `ROOT_PATH`
- Environment-driven secure configuration
- Local, file-based vector store (no network DB surface to secure)

## Operational Recommendations

- Run ingestion as a dedicated service account.
- Keep the `ANTHROPIC_API_KEY` in a vault-backed secret; do not commit it to `.env`.
- Back up the `CHROMA_PERSIST_DIR` directory (or rebuild it with a reindex from
  the source documents — it is fully reproducible from `ROOT_PATH`).
- Add retry/circuit-breaker around model calls for hardening.
- Add OpenTelemetry tracing if centralized observability is required.
- Tune `HYBRID_SEARCH_ALPHA` based on your use case:
  - Higher alpha: better for conceptual/semantic queries
  - Lower alpha: better for keyword-specific searches
```
