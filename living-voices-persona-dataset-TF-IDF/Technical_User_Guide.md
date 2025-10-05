# 📗 Technical Documentation and User Guide

## 1. Introduction
This repository provides an **Information Retrieval and RAG (Retrieval-Augmented Generation)** system, featuring:
- TF-IDF inverted index,
- Multi-model hybrid retrieval,
- Context window management,
- Result caching,
- Evaluation utilities.
---

## 2. Quick Start

### 2.1 Install dependencies
```bash
pip install -r requirements.txt
```
Optional modules: `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `torch`.
---

### 2.2 Data Preparation
| File | Columns |
|------|----------|
| `corpus.csv` | `doc_id, text` |
| `queries.csv` | `query_id, query` |
| `qrels.csv` | `query_id, doc_id, label (0/1)` |
---

### 2.3 Build Index and Basic Retrieval (TF-IDF)
```python
from tfidf_search import TfidfRetrieval
import pandas as pd

corpus = pd.read_csv("corpus.csv")
retr = TfidfRetrieval(tokenizer_mode="mixed", ngram_range=(1,2), min_df=1)
retr.fit(corpus, text_col="text", id_col="doc_id")

retr.search("Sample query: EYLF belonging growth", top_k=5)
```
---

### 2.4 RAG Pipeline Example
```python
from rag_pipeline import RAGPipeline
rag = RAGPipeline(retriever=..., reranker=..., context_window=..., cache=...)
rag.answer("Your question here", top_k=10, max_chars=1500)
```
---

### 2.5 Evaluation
**Option A – Notebook**
```python
from eval_runner import run_eval
used, metrics, table = run_eval(
    corpus_csv="corpus.csv",
    queries_csv="queries.csv",
    qrels_csv="qrels.csv",
    tokenizer_mode="mixed",
    top_k=50,
    ngram_range=(1,2),
    min_df=1
)
```

**Option B – Command Line**
```bash
python eval_runner.py   --corpus_csv corpus.csv   --queries_csv queries.csv   --qrels_csv qrels.csv   --tokenizer_mode mixed   --top_k 50
```
---

## 3. Configuration Parameters
| Parameter | Description | Example |
|------------|-------------|----------|
| `tokenizer_mode` | Tokenization mode (`"zh"`, `"en"`, `"mixed"`) | `"mixed"` |
| `ngram_range` | N-gram range | `(1, 2)` |
| `min_df` | Minimum document frequency | `1` |
| `weights` | Fusion weights for hybrid retrieval | `{"tfidf":0.4,"bm25":0.3,"sbert":0.3}` |
| `phrase_boost` | Phrase matching bonus | `True` |
| `max_chars` / `overlap_chars` | Context window length and overlap | `1500 / 150` |
| `cache_size` | LRU cache capacity | `256` |
---

## 4. API Overview
### 4.1 `TfidfRetrieval`
| Method | Description |
|--------|-------------|
| `fit(df, text_col, id_col)` | Train and build the inverted index |
| `search(query, top_k)` | Return ranked `(doc_id, score)` pairs |
| `save(path)` / `load(path)` | Persist or load the model and index |

### 4.2 `HybridRetriever`
| Method | Description |
|--------|-------------|
| `search(query, top_k, weights)` | Perform multi-model retrieval fusion |
| Dependencies | `BM25Retriever`, `SentenceBertRetriever` (optional) |

### 4.3 `ContextWindow`
| Method | Description |
|--------|-------------|
| `pack(doc_ids, max_chars, overlap_chars)` | Merge segments into contextual windows |

### 4.4 `CrossEncoderReranker` (optional)
| Method | Description |
|--------|-------------|
| `rerank(query, candidates)` | Rerank top candidates using a cross-encoder |
---

## 5. Best Practices
- For Chinese or multilingual corpora, use `tokenizer_mode="mixed"` with `(1,2)` n-grams.  
- Split long documents into smaller chunks before indexing.  
- Retrieve top 20–50 candidates and rerank the top subset.  
- Rebuild indexes periodically and monitor cache hit rate and latency.  
---

## 6. Performance and Capacity
| Metric | Example Value |
|--------|----------------|
| Index size | ~[X MB] |
| Build time | ~[X min] |
| P95 latency | ~[Y ms] (CPU / GPU) |
---

## 7. Troubleshooting (FAQ)
| Issue | Cause | Solution |
|-------|--------|-----------|
| `ModuleNotFoundError: eval_runner` | The file path is not visible to the notebook | Add `import sys; sys.path.append("/mnt/data")` or move file to working directory |
| All zero results | Tokenization or `min_df` too strict | Lower `min_df`, verify column names |
| Poor Chinese retrieval | Tokenizer issue | Enable `"mixed"` mode or add domain-specific dictionary |
---

## 8. Version and Change Log
| Version | Description |
|----------|--------------|
| v1.0 | Implemented TF-IDF engine and baseline evaluation |
| v1.1 | Added BM25, SBERT, Hybrid retrievers, and CrossEncoder reranking |
| v1.2 | Added caching and context window management |
