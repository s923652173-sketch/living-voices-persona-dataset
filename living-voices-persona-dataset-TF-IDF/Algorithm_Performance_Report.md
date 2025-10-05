# 📘 Algorithm Performance Evaluation Report

## 1. Executive Summary
**Objective:**  
Evaluate the effectiveness of the project’s retrieval and RAG modules on the **[dataset/domain name]**.

**Methodology:**  
Compare **TF-IDF**, **BM25**, **SBERT**, and **Hybrid (+Rerank)** models using metrics including **Recall@K**, **Precision@K**, **MRR@K**, **MAP@K**, and **nDCG@K**.

**Conclusion:**  
At **K = [value]**, the best-performing configuration is **[model/combination]**, achieving a **[x%]** improvement over the baseline in **nDCG@K / MAP@K**.  

**Business Impact:**  
In **[application scenario]**, the average top-1 hit rate (MRR@1) reaches **[xx]**, resulting in **shorter response time / reduced false retrievals**.

---

## 2. Dataset and Annotation
- **Corpus size:** [N] documents; [M] deduplicated paragraphs.  
- **Data source and cleaning:** Filtering, segmentation, noise removal, length normalization, and stopword removal.  
- **Query set:** [Q] queries; generated via real queries, expert construction, or historical logs.  
- **Relevance annotations (qrels):** Binary labels (0/1), annotated by [number] annotators (optional inter-rater Kappa).

**File formats (CSV):**
```
corpus.csv   → (doc_id, text)
queries.csv  → (query_id, query)
qrels.csv    → (query_id, doc_id, label)
```
---

## 3. Methods and Configuration
- **Tokenization:** `tokenizer_mode = [zh/en/mixed]`; `ngram_range = (1, 2)`; `min_df = [1/2/...]`.  
- **Model descriptions:**
  - **TF-IDF:** Cosine similarity with inverted index + vectorized search.  
  - **BM25:** Rank-BM25 implementation with parameters **(k1, b)**.  
  - **SBERT:** Sentence-BERT (*paraphrase-multilingual-MiniLM...*) embeddings with optional FAISS retrieval.  
  - **Hybrid(+Rerank):** Weighted fusion (**w_tfidf, w_bm25, w_sbert**) with optional CrossEncoder reranking.  
- **Context window:** `max_chars = [X]`, `overlap = [Y]`; deduplication applied.  
- **Cache:** LRU-based cache mechanism (optional).
---

## 4. Evaluation Setup
- **Metrics:** Recall@{1,5,10}, Precision@{1,5,10}, MRR@10, MAP@10, nDCG@10 (main metric: **nDCG@10**).  
- **Data split:** [5-fold / temporal / hold-out ratio]; fixed random seed.  
- **Environment:** CPU/GPU, memory, Python version, major dependencies.
---

## 5. Results and Visualization

### 5.1 Main Results
| Model | Recall@10 | Precision@10 | MRR@10 | MAP@10 | nDCG@10 |
|-------|-----------:|--------------:|--------:|--------:|--------:|
| TF-IDF | 0.62 | 0.21 | 0.44 | 0.33 | 0.58 |
| BM25 | 0.65 | 0.22 | 0.46 | 0.35 | 0.60 |
| SBERT | 0.68 | 0.24 | 0.49 | 0.37 | 0.63 |
| **Hybrid(+Rerank)** | **0.74** | **0.27** | **0.56** | **0.43** | **0.69** |

**Visualizations to include in the report:**
1. **nDCG@K line chart** (K = 1, 3, 5, 10)  
2. **MAP@10 bar chart** comparing all models  
*(You can capture these directly from the notebook output.)*
---

### 5.2 Ablation Study (Optional)
Remove components such as reranking or BM25, or use TF-IDF only, to measure the contribution of each component.

### 5.3 Sensitivity Analysis (Optional)
Analyze the impact of parameters like `ngram_range`, `min_df`, fusion weights, and context window length (via heatmap or line chart).
---

## 6. Error Analysis
**Representative failure cases (3–5 examples):**  
List queries, incorrectly retrieved documents, and missed relevant documents.  
Analyze possible reasons:

| Cause | Suggestion |
|--------|-------------|
| Lemmatization / synonym gap | Add synonym dictionary or query expansion |
| Overlong context diluting signal | Use finer segmentation |
| Multilingual / code-mixing issue | Use `tokenizer_mode="mixed"` or multilingual model |

Include statistical insights (e.g., document length, term density, language distribution).
---

## 7. Efficiency and Resource Usage
- Index build time, index size  
- Average query latency (P50/P95)  
- QPS throughput  
- Resource comparison: TF-IDF vs. SBERT (GPU usage if applicable)
---

## 8. Conclusions and Recommendations
- **Best configuration:** [model/Hybrid+Rerank] with **+x%** improvement on main metric.  
- **Deployment suggestions:**  
  - Deploy **Hybrid+Rerank** retriever with parallel Top-K caching.  
  - Rebuild indexes periodically (every [time period]).  
  - Continuously enhance relevance labels and query expansion lexicons.
---

## 9. Reproducibility
- Include experiment commands or notebook cell references.  
- Fix random seeds and dependency versions (see `requirements.txt`).  
- Attach data samples (`corpus.csv`, `queries.csv`, `qrels.csv`) and evaluation scripts.  
