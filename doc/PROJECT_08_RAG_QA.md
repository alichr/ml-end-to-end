# Project 8: RAG-Powered Q&A System (Hard+)

## Goal

Build a **production-grade question-answering system** that retrieves relevant passages from
a document corpus, feeds them to an LLM, and generates accurate answers **with source
citations**. The system should handle document ingestion, semantic search, answer generation,
and serve everything behind a streaming chat API with a conversational frontend.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why RAG? | Retrieval-Augmented Generation is the dominant pattern for grounding LLMs in private data. Every enterprise AI team is building some version of this. |
| Why not just fine-tune? | Fine-tuning is expensive, slow, and hallucinates proprietary data. RAG lets you swap documents without retraining. |
| Will this look good on a portfolio? | Absolutely. RAG systems are the #1 most requested skill in AI engineering job postings right now. |
| What makes this "Hard+"? | You must handle non-deterministic outputs, streaming responses, cost management, and evaluate systems where there is no single correct answer. |

---

## Architecture Overview

```
                         ┌─────────────────┐
                         │   Streamlit /    │
                         │   Chat Frontend  │
                         └────────┬────────┘
                                  │ WebSocket / SSE
                                  ▼
                         ┌─────────────────┐
                         │    FastAPI       │
                         │  Chat Endpoint   │
                         │  (streaming)     │
                         └───┬─────────┬───┘
                             │         │
                    ┌────────▼──┐  ┌───▼──────────┐
                    │  Retriever │  │  LLM API     │
                    │  (Vector   │  │  (Claude /   │
                    │   Search)  │  │   OpenAI)    │
                    └────┬───────┘  └──────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │ ChromaDB │ │  BM25  │ │Cross-Enc │
        │ (Vector) │ │(Keyword)│ │(Reranker)│
        └──────────┘ └────────┘ └──────────┘

     ┌──────────────────────────────────────────┐
     │          Document Ingestion Pipeline      │
     │  Upload → Parse → Chunk → Embed → Index  │
     └──────────────────────────────────────────┘

     ┌──────────────┐  ┌──────────────┐
     │  Prometheus   │  │   Grafana    │
     │  (token cost, │  │  (dashboard) │
     │   latency)    │  │              │
     └──────────────┘  └──────────────┘

Everything runs in Docker. LLM calls go to external APIs.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML/AI ecosystem standard |
| LLM | Claude API / OpenAI API | Best-in-class generation with function calling |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Fast, high-quality, runs locally |
| Vector Database | ChromaDB (dev) / Pinecone (prod) | ChromaDB for local dev, Pinecone for scalable prod |
| Orchestration | LangChain | Standard RAG abstractions, good for prototyping |
| API Framework | FastAPI | Async support, streaming responses, auto-docs |
| Frontend | Streamlit | Quick chat UI with streaming support |
| Containerization | Docker + docker-compose | Consistent environments across dev and prod |
| CI/CD | GitHub Actions | Free, integrated with GitHub |
| Monitoring | Prometheus + Grafana | Industry standard observability |
| Testing | pytest + RAGAS | Standard testing + RAG-specific evaluation framework |
| Document Parsing | unstructured / PyPDF2 | Handle PDF, HTML, Markdown, DOCX |

---

## Project Structure

```
rag-qa-system/
│
├── doc/
│   ├── DESIGN_DOC.md              # Scope, latency/accuracy/cost targets
│   ├── PROJECT_PLAN.md            # This file
│   └── MODEL_CARD.md              # RAG system documentation
│
├── pyproject.toml                 # Dependencies and project metadata
├── README.md                      # Setup instructions, architecture, quickstart
│
├── configs/
│   ├── rag_config.yaml            # Chunking, retrieval, generation settings
│   ├── embedding_config.yaml      # Embedding model, batch size, dimensions
│   └── serve_config.yaml          # API settings, rate limits, cost caps
│
├── data/                          # Git-ignored
│   ├── raw/                       # Original documents (PDF, HTML, MD)
│   ├── processed/                 # Parsed and cleaned text
│   └── evaluation/                # Golden Q&A pairs for evaluation
│
├── notebooks/
│   ├── 01_chunking_experiments.ipynb   # Compare chunking strategies
│   ├── 02_retrieval_evaluation.ipynb   # Evaluate retrieval quality
│   └── 03_e2e_evaluation.ipynb         # End-to-end RAG evaluation
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py              # Document parsing (PDF, HTML, MD, DOCX)
│   │   ├── chunker.py             # Chunking strategies
│   │   ├── metadata.py            # Metadata extraction
│   │   └── pipeline.py            # End-to-end ingestion pipeline
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embedder.py            # Embedding model wrapper
│   │   └── indexer.py             # Vector DB indexing
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py       # Semantic search via ChromaDB/Pinecone
│   │   ├── keyword_search.py      # BM25 keyword search
│   │   ├── hybrid.py              # Combined vector + keyword search
│   │   └── reranker.py            # Cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py    # Prompt engineering templates
│   │   ├── generator.py           # LLM call with context injection
│   │   ├── citations.py           # Extract and validate source citations
│   │   └── memory.py              # Conversation memory management
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py   # Recall@k, MRR, NDCG
│   │   ├── generation_metrics.py  # Faithfulness, relevance, hallucination
│   │   └── ragas_eval.py          # RAGAS framework integration
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   ├── schemas.py             # Request/response Pydantic models
│   │   ├── streaming.py           # SSE/WebSocket streaming logic
│   │   └── upload.py              # Document upload endpoint
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Prometheus metric definitions
│   │   ├── cost_tracker.py        # Token usage and cost tracking
│   │   └── quality_monitor.py     # Retrieval quality monitoring
│   │
│   └── frontend/
│       └── app.py                 # Streamlit chat UI
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py        # Chunking correctness
│   │   ├── test_citations.py      # Citation extraction
│   │   ├── test_schemas.py        # API schema validation
│   │   └── test_prompt_templates.py
│   ├── integration/
│   │   ├── test_ingestion.py      # Full ingestion pipeline
│   │   ├── test_retrieval.py      # Retrieval quality checks
│   │   ├── test_api.py            # Full API pipeline
│   │   └── test_e2e_rag.py        # End-to-end RAG pipeline
│   ├── golden/
│   │   └── golden_qa_pairs.json   # Curated Q&A for regression testing
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.api             # API server
│   ├── Dockerfile.frontend        # Streamlit container
│   └── Dockerfile.ingestion       # Document processing worker
│
├── docker-compose.yaml            # Orchestrate all services
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                # Lint → Test → Build on PR
│       └── cd.yaml                # Deploy on merge to main
│
├── grafana/
│   └── dashboards/
│       └── rag_monitoring.json    # Cost, latency, quality dashboard
│
├── prometheus/
│   └── prometheus.yml
│
└── scripts/
    ├── setup.sh                   # One-command project setup
    ├── ingest.sh                  # Run document ingestion
    └── evaluate.sh                # Run evaluation suite
```

---

## Phase 1: Setup & Design Doc

**Duration:** 2 days
**Objective:** Define scope, targets, and constraints before writing any code.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given a question and a corpus of documents, return an accurate
     answer with source citations"
   - **Success criteria:**
     - Retrieval recall@5 >= 85% on evaluation set
     - Answer faithfulness >= 90% (no hallucinated facts)
     - End-to-end latency < 5 seconds (non-streaming), first token < 1 second (streaming)
     - Cost per query < $0.02 average
     - Support corpus up to 10,000 documents
   - **Out of scope:** multi-language, real-time document updates, image/table understanding
   - **Risks:** hallucination, prompt injection, cost overruns, embedding model drift

2. **Initialize the repository**
   - `git init`, create `.gitignore` (data/, *.db, .env, __pycache__/, chromadb/)
   - Set up branch strategy: `main`, `dev`, `feature/*`
   - Create `pyproject.toml` with all dependencies:
     ```toml
     [project]
     name = "rag-qa-system"
     dependencies = [
         "anthropic>=0.20.0",
         "openai>=1.10.0",
         "sentence-transformers>=2.3.0",
         "chromadb>=0.4.0",
         "langchain>=0.1.0",
         "langchain-community>=0.0.10",
         "fastapi>=0.109.0",
         "uvicorn>=0.27.0",
         "streamlit>=1.30.0",
         "unstructured>=0.12.0",
         "pypdf2>=3.0.0",
         "ragas>=0.1.0",
         "rank-bm25>=0.2.2",
         "prometheus-client>=0.20.0",
         "pydantic>=2.5.0",
         "pydantic-settings>=2.1.0",
         "python-multipart>=0.0.6",
         "sse-starlette>=1.8.0",
     ]
     ```

3. **Create the folder structure** (as shown above)

4. **Set up development environment**
   - Python virtual environment
   - Pre-commit hooks: ruff, mypy
   - Environment variable management for API keys (`.env` file, never committed)
   - Verify LLM API access: test a simple Claude/OpenAI call

### Skills Learned

- Designing RAG systems (defining scope and non-functional requirements)
- Cost budgeting for LLM applications
- Secure API key management

---

## Phase 2: Document Pipeline

**Duration:** 4-5 days
**Objective:** Build a robust pipeline that ingests documents, chunks them, and extracts metadata.

### Tasks

1. **Document parsing** -- `src/ingestion/parser.py`
   - Support multiple formats: PDF, HTML, Markdown, plain text, DOCX
   - Use `unstructured` library for intelligent parsing
   - Handle edge cases: scanned PDFs (OCR fallback), tables, headers/footers
   - Preserve document structure (headings, paragraphs, lists)
   ```python
   class DocumentParser:
       """Parse documents from multiple formats into structured text."""

       def parse(self, file_path: Path) -> ParsedDocument:
           suffix = file_path.suffix.lower()
           parser_map = {
               ".pdf": self._parse_pdf,
               ".html": self._parse_html,
               ".md": self._parse_markdown,
               ".txt": self._parse_text,
               ".docx": self._parse_docx,
           }
           if suffix not in parser_map:
               raise UnsupportedFormatError(f"Format {suffix} not supported")
           return parser_map[suffix](file_path)

       def _parse_pdf(self, file_path: Path) -> ParsedDocument:
           elements = partition_pdf(str(file_path))
           return ParsedDocument(
               content="\n\n".join(str(el) for el in elements),
               metadata={"source": file_path.name, "format": "pdf"},
           )
   ```

2. **Chunking strategies** -- `src/ingestion/chunker.py`
   - Implement three strategies and compare them:
     - **Fixed-size:** split every N tokens with overlap
     - **Semantic:** split at paragraph/section boundaries
     - **Recursive:** LangChain's `RecursiveCharacterTextSplitter`
   - Each chunk must carry metadata: source document, page number, chunk index
   - Target chunk size: 500-1000 tokens (experiment to find optimal)
   ```python
   class RecursiveChunker:
       """Recursively split text by decreasing separator priority."""

       def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
           self.splitter = RecursiveCharacterTextSplitter(
               chunk_size=chunk_size,
               chunk_overlap=chunk_overlap,
               separators=["\n\n", "\n", ". ", " ", ""],
               length_function=self._token_length,
           )

       def chunk(self, document: ParsedDocument) -> list[Chunk]:
           texts = self.splitter.split_text(document.content)
           return [
               Chunk(
                   text=text,
                   metadata={
                       **document.metadata,
                       "chunk_index": i,
                       "chunk_size_tokens": self._token_length(text),
                   },
               )
               for i, text in enumerate(texts)
           ]
   ```

3. **Metadata extraction** -- `src/ingestion/metadata.py`
   - Extract: title, author, date, section headings, page numbers
   - Generate a summary of each document (optional LLM call)
   - Tag chunks with their parent section heading for better retrieval context

4. **Ingestion pipeline** -- `src/ingestion/pipeline.py`
   - End-to-end: directory of documents --> parsed --> chunked --> metadata-enriched
   - Idempotent: re-running on the same documents should not create duplicates
   - Progress tracking for large corpus ingestion
   - Error handling: skip corrupted files, log warnings, continue processing

5. **Chunking experiments** -- `notebooks/01_chunking_experiments.ipynb`
   - Compare chunk sizes: 256, 512, 1024 tokens
   - Compare strategies: fixed vs semantic vs recursive
   - Evaluate: average chunk coherence, retrieval performance on sample queries
   - Visualize chunk size distributions

### Skills Learned

- Document parsing across formats (a real-world pain point)
- Chunking strategies and their trade-offs
- Building idempotent data pipelines
- Metadata engineering for retrieval

---

## Phase 3: Embedding & Indexing

**Duration:** 3-4 days
**Objective:** Convert text chunks into vectors and build a searchable index.

### Tasks

1. **Choose and configure embedding model** -- `src/embedding/embedder.py`
   - Start with `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)
   - Wrap in a class that handles batching and GPU/CPU selection
   - Benchmark alternatives: `all-mpnet-base-v2` (768d, higher quality, slower)
   ```python
   class EmbeddingModel:
       """Wrapper around sentence-transformers for document embedding."""

       def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
           self.model = SentenceTransformer(model_name)
           self.dimension = self.model.get_sentence_embedding_dimension()

       def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
           return self.model.encode(
               texts,
               batch_size=batch_size,
               show_progress_bar=True,
               normalize_embeddings=True,
           )

       def embed_query(self, query: str) -> np.ndarray:
           return self.model.encode(
               query, normalize_embeddings=True
           )
   ```

2. **Build vector index** -- `src/embedding/indexer.py`
   - ChromaDB for local development (persistent storage, no server needed)
   - Create collection with appropriate distance metric (cosine similarity)
   - Batch upsert with metadata
   - Handle index updates: add new documents, delete removed ones
   ```python
   class VectorIndexer:
       def __init__(self, collection_name: str = "documents"):
           self.client = chromadb.PersistentClient(path="./chromadb_data")
           self.collection = self.client.get_or_create_collection(
               name=collection_name,
               metadata={"hnsw:space": "cosine"},
           )

       def index_chunks(self, chunks: list[Chunk], embeddings: np.ndarray):
           self.collection.upsert(
               ids=[chunk.id for chunk in chunks],
               embeddings=embeddings.tolist(),
               documents=[chunk.text for chunk in chunks],
               metadatas=[chunk.metadata for chunk in chunks],
           )
   ```

3. **Evaluate retrieval quality** -- `notebooks/02_retrieval_evaluation.ipynb`
   - Create 50+ evaluation queries with known relevant documents
   - Measure: Recall@1, Recall@5, Recall@10, MRR, NDCG
   - Compare embedding models side by side
   - Analyze failure cases: what queries retrieve irrelevant chunks?

4. **Optimize indexing performance**
   - Benchmark ingestion throughput (documents per second)
   - Tune HNSW parameters for recall vs speed trade-off
   - Implement incremental indexing (only embed new/changed documents)

### Skills Learned

- Embedding model selection and evaluation
- Vector database operations (CRUD, search, filtering)
- Retrieval evaluation metrics (Recall@k, MRR, NDCG)
- Index performance tuning

---

## Phase 4: RAG Pipeline

**Duration:** 5-6 days
**Objective:** Build the core retrieve-then-generate pipeline with citations.

### Tasks

1. **Retrieval module** -- `src/retrieval/vector_search.py`
   - Query the vector index, return top-k chunks with scores
   - Apply metadata filters (e.g., filter by document source, date range)
   - Handle empty results gracefully
   ```python
   class VectorRetriever:
       def retrieve(
           self, query: str, top_k: int = 5, filters: dict | None = None
       ) -> list[RetrievedChunk]:
           query_embedding = self.embedder.embed_query(query)
           results = self.collection.query(
               query_embeddings=[query_embedding.tolist()],
               n_results=top_k,
               where=filters,
               include=["documents", "metadatas", "distances"],
           )
           return [
               RetrievedChunk(
                   text=doc,
                   metadata=meta,
                   score=1 - dist,  # Convert distance to similarity
               )
               for doc, meta, dist in zip(
                   results["documents"][0],
                   results["metadatas"][0],
                   results["distances"][0],
               )
           ]
   ```

2. **Prompt engineering** -- `src/generation/prompt_templates.py`
   - Design system prompt that instructs the LLM to:
     - Only answer from provided context
     - Cite sources using [Source: filename, page X] format
     - Say "I don't know" when context is insufficient
   - Template for multi-turn conversations with history
   ```python
   SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on
   the provided context documents. Follow these rules strictly:

   1. ONLY use information from the provided context to answer
   2. For every claim, cite the source using [Source: <filename>, p.<page>]
   3. If the context does not contain enough information, say:
      "I don't have enough information in the provided documents to answer this."
   4. Never make up or hallucinate information
   5. If multiple sources agree, cite all of them
   6. Be concise but thorough

   Context documents:
   {context}
   """

   USER_TEMPLATE = """Question: {question}

   Please answer based on the context above, with source citations."""
   ```

3. **Generation with streaming** -- `src/generation/generator.py`
   - Call LLM API with context-stuffed prompt
   - Stream response tokens back to the caller
   - Manage context window: if retrieved chunks exceed token limit, truncate
   - Implement fallback: if primary LLM fails, try secondary
   ```python
   class RAGGenerator:
       async def generate_stream(
           self, query: str, context_chunks: list[RetrievedChunk]
       ) -> AsyncGenerator[str, None]:
           context = self._format_context(context_chunks)
           messages = [
               {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
               {"role": "user", "content": USER_TEMPLATE.format(question=query)},
           ]

           # Ensure we stay within context window
           total_tokens = self._count_tokens(messages)
           if total_tokens > self.max_context_tokens:
               context_chunks = self._truncate_context(
                   context_chunks, self.max_context_tokens
               )

           async with self.client.messages.stream(
               model=self.model_name,
               messages=messages,
               max_tokens=1024,
           ) as stream:
               async for text in stream.text_stream:
                   yield text
   ```

4. **Citation extraction and validation** -- `src/generation/citations.py`
   - Parse citations from generated text: `[Source: filename, p.X]`
   - Validate that cited sources were actually in the provided context
   - Flag answers that make claims without citations
   - Return structured citation objects alongside the answer

5. **Conversation memory** -- `src/generation/memory.py`
   - Maintain conversation history per session
   - Implement sliding window: keep last N turns
   - Condense old conversation into a summary to save tokens
   - Handle follow-up questions that reference previous context
   ```python
   class ConversationMemory:
       def __init__(self, max_turns: int = 10):
           self.max_turns = max_turns
           self.history: list[dict] = []

       def add_turn(self, role: str, content: str):
           self.history.append({"role": role, "content": content})
           if len(self.history) > self.max_turns * 2:
               self._condense_history()

       def get_context_messages(self) -> list[dict]:
           return self.history.copy()
   ```

6. **Context window management**
   - Calculate token usage: system prompt + context + history + query
   - Prioritize: most relevant chunks first, then history
   - Implement dynamic context sizing based on query complexity

### Skills Learned

- Prompt engineering for grounded generation
- Context window management (a critical production skill)
- Streaming response patterns
- Citation extraction from LLM output
- Conversation memory strategies

---

## Phase 5: Evaluation

**Duration:** 4-5 days
**Objective:** Rigorously evaluate both retrieval and generation quality.

### Tasks

1. **Build evaluation dataset**
   - Create 100+ question-answer-context triples manually
   - Categories: factual lookup, multi-hop reasoning, unanswerable questions
   - Store as structured JSON in `data/evaluation/golden_qa_pairs.json`
   ```json
   {
     "questions": [
       {
         "id": "q001",
         "question": "What is the maximum penalty for GDPR violations?",
         "expected_answer": "Up to 20 million euros or 4% of global annual turnover",
         "relevant_documents": ["gdpr_overview.pdf"],
         "relevant_chunks": ["chunk_42", "chunk_43"],
         "category": "factual_lookup"
       }
     ]
   }
   ```

2. **Retrieval metrics** -- `src/evaluation/retrieval_metrics.py`
   - **Recall@k:** what fraction of relevant chunks appear in top-k results?
   - **MRR (Mean Reciprocal Rank):** how high is the first relevant result?
   - **NDCG:** does the ranking order match the ideal order?
   - Evaluate across query categories (factual, multi-hop, unanswerable)
   ```python
   def recall_at_k(
       retrieved_ids: list[str], relevant_ids: list[str], k: int
   ) -> float:
       retrieved_set = set(retrieved_ids[:k])
       relevant_set = set(relevant_ids)
       return len(retrieved_set & relevant_set) / len(relevant_set)

   def mean_reciprocal_rank(
       retrieved_ids: list[str], relevant_ids: list[str]
   ) -> float:
       relevant_set = set(relevant_ids)
       for rank, doc_id in enumerate(retrieved_ids, 1):
           if doc_id in relevant_set:
               return 1.0 / rank
       return 0.0
   ```

3. **Generation quality metrics** -- `src/evaluation/generation_metrics.py`
   - **Faithfulness:** does the answer only use information from the context?
   - **Answer relevance:** does the answer actually address the question?
   - **Hallucination rate:** percentage of claims not grounded in context
   - **Citation accuracy:** are citations correct and complete?
   - Use LLM-as-judge for automated evaluation
   ```python
   async def evaluate_faithfulness(
       answer: str, context: str, judge_client: Any
   ) -> float:
       """Use an LLM to judge whether the answer is faithful to context."""
       prompt = f"""Given the following context and answer, rate the faithfulness
       of the answer on a scale of 0-1. Faithfulness means every claim in the
       answer is supported by the context.

       Context: {context}
       Answer: {answer}

       Return only a number between 0 and 1."""

       response = await judge_client.complete(prompt)
       return float(response.strip())
   ```

4. **RAGAS framework integration** -- `src/evaluation/ragas_eval.py`
   - Integrate the RAGAS library for standardized RAG evaluation
   - Metrics: faithfulness, answer relevancy, context precision, context recall
   - Generate evaluation reports with per-question breakdowns

5. **End-to-end evaluation notebook** -- `notebooks/03_e2e_evaluation.ipynb`
   - Run full evaluation pipeline on golden dataset
   - Visualize: retrieval recall by category, faithfulness distribution
   - Error analysis: which question types fail? why?
   - Compare configurations: chunk size, top-k, embedding model
   - Document findings and chosen configuration

6. **Regression testing setup**
   - Define a "golden set" of 20 critical Q&A pairs
   - Automated test that runs on every PR: retrieval recall must not drop
   - Alert if answer quality degrades beyond threshold

### Skills Learned

- Evaluating retrieval systems (IR metrics)
- Evaluating generative systems (faithfulness, hallucination)
- LLM-as-judge evaluation patterns
- Building regression test suites for non-deterministic systems
- RAGAS framework for standardized RAG evaluation

---

## Phase 6: Advanced Features

**Duration:** 4-5 days
**Objective:** Improve retrieval and generation quality with advanced techniques.

### Tasks

1. **Hybrid search** -- `src/retrieval/hybrid.py`
   - Combine vector search (semantic) with BM25 (keyword matching)
   - Reciprocal Rank Fusion (RRF) to merge rankings
   - This catches cases where exact terms matter (e.g., product names, IDs)
   ```python
   class HybridRetriever:
       def __init__(self, vector_retriever, keyword_retriever, alpha: float = 0.7):
           self.vector = vector_retriever
           self.keyword = keyword_retriever
           self.alpha = alpha  # Weight for vector results

       def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
           vector_results = self.vector.retrieve(query, top_k=top_k * 2)
           keyword_results = self.keyword.retrieve(query, top_k=top_k * 2)
           return self._reciprocal_rank_fusion(
               vector_results, keyword_results, top_k
           )

       def _reciprocal_rank_fusion(
           self, results_a, results_b, top_k, k=60
       ) -> list[RetrievedChunk]:
           scores = {}
           for rank, chunk in enumerate(results_a):
               scores[chunk.id] = self.alpha / (k + rank + 1)
           for rank, chunk in enumerate(results_b):
               scores[chunk.id] = scores.get(chunk.id, 0) + (1 - self.alpha) / (k + rank + 1)
           sorted_ids = sorted(scores, key=scores.get, reverse=True)
           return [self._get_chunk(cid) for cid in sorted_ids[:top_k]]
   ```

2. **BM25 keyword search** -- `src/retrieval/keyword_search.py`
   - Build BM25 index over all chunks using `rank_bm25`
   - Tokenize with simple whitespace + lowercasing
   - This complements vector search for exact-match queries

3. **Cross-encoder reranking** -- `src/retrieval/reranker.py`
   - After initial retrieval (top 20), rerank with a cross-encoder
   - Use `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed
   - Cross-encoders are more accurate than bi-encoders but too slow for first-stage retrieval
   ```python
   class CrossEncoderReranker:
       def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
           self.model = CrossEncoder(model_name)

       def rerank(
           self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
       ) -> list[RetrievedChunk]:
           pairs = [(query, chunk.text) for chunk in chunks]
           scores = self.model.predict(pairs)
           ranked = sorted(
               zip(chunks, scores), key=lambda x: x[1], reverse=True
           )
           return [chunk for chunk, score in ranked[:top_k]]
   ```

4. **Query decomposition**
   - For complex multi-hop questions, break into sub-queries
   - Use LLM to decompose: "Compare X and Y" --> ["What is X?", "What is Y?"]
   - Retrieve separately for each sub-query, then merge contexts
   - Answer using the combined context

5. **Re-evaluate with advanced features**
   - Re-run evaluation suite from Phase 5
   - Compare: basic vector search vs hybrid vs hybrid + reranking
   - Expected: +5-15% improvement in retrieval recall
   - Document which features provide the most lift

### Skills Learned

- Hybrid search (vector + keyword) -- essential for production RAG
- Cross-encoder reranking -- the secret weapon for retrieval quality
- Query decomposition for complex questions
- Reciprocal Rank Fusion for combining multiple retrievers

---

## Phase 7: API & Serving

**Duration:** 4-5 days
**Objective:** Build a production-quality API with streaming chat and document upload.

### Tasks

1. **Define API schemas** -- `src/serving/schemas.py`
   ```python
   class ChatRequest(BaseModel):
       question: str
       session_id: str | None = None  # For conversation continuity
       top_k: int = 5
       filters: dict | None = None    # Metadata filters

   class ChatResponse(BaseModel):
       answer: str
       citations: list[Citation]
       retrieved_chunks: list[ChunkSummary]
       session_id: str
       token_usage: TokenUsage
       latency_ms: float

   class Citation(BaseModel):
       source: str          # Document filename
       page: int | None     # Page number if available
       chunk_text: str      # The supporting text snippet
       relevance_score: float

   class TokenUsage(BaseModel):
       prompt_tokens: int
       completion_tokens: int
       total_tokens: int
       estimated_cost_usd: float

   class DocumentUploadResponse(BaseModel):
       document_id: str
       filename: str
       num_chunks: int
       status: str          # "processed" or "failed"
       processing_time_ms: float
   ```

2. **Build FastAPI application** -- `src/serving/app.py`
   - `POST /chat` -- ask a question, get a complete answer with citations
   - `POST /chat/stream` -- ask a question, get a streaming response via SSE
   - `POST /documents/upload` -- upload a new document to the corpus
   - `DELETE /documents/{doc_id}` -- remove a document from the corpus
   - `GET /documents` -- list all indexed documents
   - `GET /health` -- health check with component status
   - `GET /metrics` -- Prometheus metrics endpoint

3. **Streaming implementation** -- `src/serving/streaming.py`
   - Server-Sent Events (SSE) for streaming responses
   - Send chunks of generated text as they arrive from the LLM
   - Send citations as a final event after generation completes
   ```python
   @app.post("/chat/stream")
   async def chat_stream(request: ChatRequest):
       async def event_generator():
           # Retrieve relevant chunks
           chunks = retriever.retrieve(request.question, top_k=request.top_k)
           yield {"event": "retrieval", "data": json.dumps({"num_chunks": len(chunks)})}

           # Stream generated answer
           full_answer = ""
           async for token in generator.generate_stream(request.question, chunks):
               full_answer += token
               yield {"event": "token", "data": token}

           # Send citations at the end
           citations = extract_citations(full_answer, chunks)
           yield {"event": "citations", "data": json.dumps([c.dict() for c in citations])}
           yield {"event": "done", "data": ""}

       return EventSourceResponse(event_generator())
   ```

4. **Document upload endpoint** -- `src/serving/upload.py`
   - Accept PDF, HTML, MD, TXT, DOCX files
   - Validate file type and size (max 50MB)
   - Process asynchronously: parse --> chunk --> embed --> index
   - Return processing status and chunk count

5. **Build Streamlit chat UI** -- `src/frontend/app.py`
   - Chat interface with message history
   - Display streaming responses in real-time
   - Show source citations as expandable sections below each answer
   - Document upload sidebar
   - Display token usage and cost per query
   - Session management (new chat, continue conversation)

6. **Error handling and edge cases**
   - Handle LLM API timeouts and rate limits with retries
   - Graceful degradation: if reranker fails, fall back to basic retrieval
   - Input validation: reject empty queries, sanitize inputs
   - Handle concurrent sessions with proper isolation

### Skills Learned

- Building streaming APIs with Server-Sent Events
- Document upload and processing endpoints
- Chat UI development with real-time streaming
- Error handling and graceful degradation patterns
- Session management for conversational systems

---

## Phase 8: Containerization

**Duration:** 2-3 days
**Objective:** Package everything into Docker containers for consistent deployment.

### Tasks

1. **API Dockerfile** -- `docker/Dockerfile.api`
   - Multi-stage build for smaller image size
   - Include embedding model in the image (or download at startup)
   - Configure ChromaDB persistent storage via volume mount
   ```dockerfile
   FROM python:3.11-slim AS builder
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install --no-cache-dir .

   FROM python:3.11-slim
   WORKDIR /app
   COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
   COPY --from=builder /usr/local/bin /usr/local/bin
   COPY src/ /app/src/
   COPY configs/ /app/configs/

   # Pre-download embedding model at build time
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

   EXPOSE 8000
   CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Frontend Dockerfile** -- `docker/Dockerfile.frontend`
   - Simple single-stage build for Streamlit
   - Configure API URL via environment variable

3. **docker-compose.yaml** -- orchestrate all services
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       volumes:
         - chromadb_data:/app/chromadb_data
       environment:
         - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
         - EMBEDDING_MODEL=all-MiniLM-L6-v2
         - CHROMA_PERSIST_DIR=/app/chromadb_data
         - LOG_LEVEL=info

     frontend:
       build:
         context: .
         dockerfile: docker/Dockerfile.frontend
       ports: ["8501:8501"]
       environment:
         - API_URL=http://api:8000
       depends_on: [api]

     prometheus:
       image: prom/prometheus
       volumes: ["./prometheus:/etc/prometheus"]
       ports: ["9090:9090"]

     grafana:
       image: grafana/grafana
       volumes: ["./grafana:/etc/grafana/provisioning"]
       ports: ["3000:3000"]
       depends_on: [prometheus]

   volumes:
     chromadb_data:
   ```

4. **Verify the full stack locally**
   - `docker compose up` -- all services start and connect
   - Upload a test document, ask questions, verify citations
   - Check Prometheus scrapes metrics, Grafana shows dashboard

### Skills Learned

- Docker for LLM applications (embedding model caching)
- Volume management for persistent vector databases
- Secrets management in Docker (API keys)
- Multi-service orchestration

---

## Phase 9: Testing & CI/CD

**Duration:** 3-4 days
**Objective:** Build a testing strategy for non-deterministic LLM systems.

### Tasks

1. **Unit tests**
   ```
   test_chunker.py
   ├── test_fixed_chunker_respects_size_limit
   ├── test_recursive_chunker_preserves_paragraphs
   ├── test_chunk_overlap_is_correct
   └── test_metadata_propagation

   test_citations.py
   ├── test_extract_citations_from_text
   ├── test_validate_citation_against_context
   └── test_handle_malformed_citations

   test_schemas.py
   ├── test_chat_response_validates_token_usage
   ├── test_citation_model_fields
   └── test_upload_response_fields
   ```

2. **Integration tests**
   ```
   test_ingestion.py
   ├── test_pdf_ingestion_end_to_end
   ├── test_markdown_ingestion_end_to_end
   └── test_duplicate_document_handling

   test_retrieval.py
   ├── test_vector_search_returns_relevant_chunks
   ├── test_hybrid_search_improves_over_vector_only
   └── test_metadata_filtering

   test_api.py
   ├── test_chat_endpoint_returns_citations
   ├── test_stream_endpoint_sends_sse_events
   ├── test_document_upload_and_query
   └── test_health_endpoint
   ```

3. **LLM-specific testing strategies**
   - **Golden dataset regression:** run 20 critical Q&A pairs on every PR
   - **Assertion-based tests:** check structural properties (citations present, answer
     length reasonable, no profanity)
   - **Mock LLM for unit tests:** replace API calls with deterministic responses
   - **Cost guards:** test that a single query never exceeds $0.10
   ```python
   @pytest.fixture
   def mock_llm():
       """Replace LLM with deterministic mock for unit tests."""
       class MockLLM:
           async def generate(self, messages):
               return "Based on the context [Source: test.pdf, p.1], the answer is 42."
       return MockLLM()

   def test_citation_extraction_with_mock(mock_llm):
       answer = mock_llm.generate([...])
       citations = extract_citations(answer)
       assert len(citations) == 1
       assert citations[0].source == "test.pdf"
   ```

4. **CI pipeline** -- `.github/workflows/ci.yaml`
   ```yaml
   name: CI
   on: [pull_request]
   jobs:
     lint:
       steps:
         - run: ruff check .
         - run: mypy src/
     unit-test:
       steps:
         - run: pytest tests/unit/ -v
     integration-test:
       steps:
         - run: pytest tests/integration/ -v --timeout=60
     golden-set:
       steps:
         - run: pytest tests/golden/ -v --timeout=120
       env:
         ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
     build:
       steps:
         - run: docker build -f docker/Dockerfile.api .
   ```

5. **CD pipeline** -- `.github/workflows/cd.yaml`
   - Build and push Docker image on merge to main
   - Run smoke test against deployed endpoint
   - Alert on golden set regression

### Skills Learned

- Testing non-deterministic LLM systems
- Mock strategies for LLM APIs
- Golden dataset regression testing
- Cost guard tests
- CI/CD for AI applications

---

## Phase 10: Monitoring & Cost Management

**Duration:** 3-4 days
**Objective:** Track system health, answer quality, and costs in production.

### Tasks

1. **Token usage and cost tracking** -- `src/monitoring/cost_tracker.py`
   - Track per-query: prompt tokens, completion tokens, total cost
   - Track per-session: cumulative tokens, cumulative cost
   - Track per-day: total spend, queries served, average cost per query
   - Set spending alerts: daily cap, per-query cap
   ```python
   class CostTracker:
       # Pricing per 1K tokens (example for Claude 3.5 Sonnet)
       PRICING = {
           "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
           "gpt-4-turbo": {"input": 0.01, "output": 0.03},
       }

       def calculate_cost(
           self, model: str, prompt_tokens: int, completion_tokens: int
       ) -> float:
           pricing = self.PRICING[model]
           return (
               prompt_tokens / 1000 * pricing["input"]
               + completion_tokens / 1000 * pricing["output"]
           )
   ```

2. **Prometheus metrics** -- `src/monitoring/metrics.py`
   - `rag_queries_total` -- counter by status (success/error)
   - `rag_latency_seconds` -- histogram (retrieval, generation, total)
   - `rag_tokens_used_total` -- counter by type (prompt/completion)
   - `rag_cost_usd_total` -- counter of total spend
   - `rag_retrieval_chunks_returned` -- histogram
   - `rag_answer_citations_count` -- histogram
   - `rag_confidence_score` -- histogram of retrieval similarity scores

3. **Retrieval quality monitoring** -- `src/monitoring/quality_monitor.py`
   - Track average retrieval similarity scores over time
   - Alert if scores drop (indicates embedding drift or corpus changes)
   - Log queries with zero relevant results (potential gaps in corpus)
   - Track citation accuracy over sampled queries

4. **Grafana dashboard** -- `grafana/dashboards/rag_monitoring.json`
   - **Row 1 -- Traffic:** query rate, error rate, active sessions
   - **Row 2 -- Latency:** retrieval latency, generation latency, total latency (p50/p95/p99)
   - **Row 3 -- Cost:** cost per query, daily spend, cumulative spend, tokens per query
   - **Row 4 -- Quality:** retrieval scores, citations per answer, zero-result queries
   - **Row 5 -- System:** CPU, memory, ChromaDB collection size

5. **Alerting rules**
   - Daily spend > $50 --> alert (possible runaway usage)
   - Average retrieval score < 0.3 for 10 minutes --> alert (quality degradation)
   - Error rate > 5% for 5 minutes --> alert
   - p95 latency > 10 seconds for 5 minutes --> alert
   - Zero-result query rate > 20% --> alert (corpus gap)

### Skills Learned

- Cost management for LLM applications (a critical production skill)
- Token usage monitoring and budgeting
- Retrieval quality monitoring
- Building LLM-specific observability dashboards
- Setting meaningful alerts for AI systems

---

## Timeline Summary

```
Week 1  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 1: Setup & Design Doc     (2 days)
        Phase 2: Document Pipeline      (3 days)

Week 2  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 2: Document Pipeline      (2 days)
        Phase 3: Embedding & Indexing   (3 days)

Week 3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 3: Embedding & Indexing   (1 day)
        Phase 4: RAG Pipeline           (4 days)

Week 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 4: RAG Pipeline           (2 days)
        Phase 5: Evaluation             (3 days)

Week 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 5: Evaluation             (2 days)
        Phase 6: Advanced Features      (3 days)

Week 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 6: Advanced Features      (2 days)
        Phase 7: API & Serving          (3 days)

Week 7  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 7: API & Serving          (2 days)
        Phase 8: Containerization       (3 days)

Week 8  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 9: Testing & CI/CD        (4 days)
        Buffer                          (1 day)

Week 9  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 10: Monitoring & Cost     (4 days)
        Buffer / polish                 (1 day)
```

**Total: ~40 days (8-9 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Designing RAG architectures (retrieval + generation pipelines)
- [ ] Document parsing across multiple formats (PDF, HTML, MD, DOCX)
- [ ] Text chunking strategies (fixed, semantic, recursive)
- [ ] Embedding models (sentence-transformers) and evaluation
- [ ] Vector databases (ChromaDB / Pinecone)
- [ ] Semantic search and similarity scoring
- [ ] LLM API integration (Claude / OpenAI)
- [ ] Prompt engineering for grounded generation
- [ ] Context window management and token budgeting
- [ ] Citation extraction and validation
- [ ] Conversation memory management
- [ ] Hybrid search (vector + BM25 keyword)
- [ ] Cross-encoder reranking
- [ ] Query decomposition for complex questions
- [ ] RAG evaluation metrics (Recall@k, MRR, NDCG)
- [ ] Generation quality evaluation (faithfulness, hallucination rate)
- [ ] RAGAS evaluation framework
- [ ] LLM-as-judge evaluation patterns
- [ ] Streaming APIs with Server-Sent Events
- [ ] Building chat UIs with Streamlit
- [ ] Testing non-deterministic LLM systems
- [ ] Golden dataset regression testing
- [ ] Token usage and cost tracking
- [ ] LLM cost management and budgeting
- [ ] Docker for LLM applications
- [ ] CI/CD for AI systems
- [ ] Monitoring and observability for RAG systems
- [ ] Retrieval quality monitoring in production

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Initialize the project
mkdir rag-qa-system && cd rag-qa-system
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,evaluation} notebooks \
  src/{ingestion,embedding,retrieval,generation,evaluation,serving,monitoring,frontend} \
  tests/{unit,integration,golden} docker .github/workflows grafana/dashboards prometheus scripts

# 3. Set up your API keys
echo "ANTHROPIC_API_KEY=your-key-here" > .env
echo ".env" >> .gitignore

# 4. Start writing DESIGN_DOC.md
```
