# Enterprise RAG System Architecture & Technical Specifications

This document provides a comprehensive technical breakdown of the **Production RAG System** architecture, data pipelines, search algorithms, mathematical foundations, and trade-offs.

---

## 1. High-Level Architectural Blueprint

The system separates concerns across three primary stages: **Ingestion**, **Multi-Stage Retrieval & Reranking**, and **Guardrailed Generation & Verification**.

```
+-----------------------------------------------------------------------------------+
|                              1. INGESTION PIPELINE                                |
|                                                                                   |
|  [PDF / TXT / Markdown] ---> [DocumentLoader] ---> [Recursive / Semantic Chunker] |
|                                                             |                     |
|                     +---------------------------------------+                     |
|                     |                                       |                     |
|                     v                                       v                     |
|          [Dense Embedder Engine]                    [BM25 Tokenizer]              |
|        (all-MiniLM-L6-v2 / BGE)                 (Okapi k1=1.5, b=0.75)            |
|                     |                                       |                     |
|                     v                                       v                     |
|           [Vector Store Matrix]                    [Inverted Index]               |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        2. RETRIEVAL & RERANKING PIPELINE                          |
|                                                                                   |
|   User Query ---> [Intent Router] ---> [HyDE / Query Expansion (Optional)]        |
|                          |                                                        |
|             +------------+-------------+                                          |
|             |                          |                                          |
|             v                          v                                          |
|    [Dense Vector Search]       [Sparse BM25 Search]                               |
|        (Cosine Top-K)              (Lexical Top-K)                                |
|             |                          |                                          |
|             +------------+-------------+                                          |
|                          |                                                        |
|                          v                                                        |
|            [Reciprocal Rank Fusion (RRF)]                                         |
|                 Score = SUM(1 / (k + rank))                                       |
|                          |                                                        |
|                          v                                                        |
|            [Cross-Encoder Reranker]                                               |
|        (ms-marco-MiniLM-L-6-v2 Cross-Attention)                                   |
|                          |                                                        |
|                          v (Filtered Top-N Chunks)                                |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        3. GENERATION & EVALUATION PIPELINE                        |
|                                                                                   |
|   [Prompt Synthesizer] ---> [LLM Generation Engine] ---> [Citation Extractor]     |
|   - Context Blocks          - Local / Groq / OpenAI      - Source Attributions    |
|   - Anti-Hallucination      - Temperature: 0.1           - Sentence Mapping       |
|                                                                                   |
|                                                                                   |
|                                  v                                                |
|                      [RAG Triad Verification]                                     |
|                      - Faithfulness Metric                                        |
|                      - Answer Relevance Score                                     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Ingestion & Chunking Strategies

### 2.1 The Chunking Dilemma
Chunk size controls the fundamental trade-off between **retrieval granularity** and **synthesizing context**:
- **Too Small (<150 tokens)**: High semantic specificity, but truncates necessary context, forcing the LLM to guess.
- **Too Large (>1000 tokens)**: Captures complete thoughts, but dilutes vector embeddings with multiple concepts ("needle in a haystack" problem).

### 2.2 Implemented Chunkers:
1. **RecursiveCharacterChunker**: Iteratively evaluates split points using a semantic separator hierarchy: `["\n\n", "\n", ". ", "? ", "! ", " ", ""]`. Ensures paragraphs and complete sentences remain unbroken.
2. **SemanticChunker**: Computes dense embeddings for consecutive sentences and tracks cosine distance. Splits text where semantic shift exceeds a percentile threshold.
3. **HierarchicalChunker (Parent-Child)**: Generates small child chunks (~250-300 chars) mapped to large parent chunks (~1200-1500 chars). Vector search indexes the child for high-precision retrieval, while the pipeline supplies the parent chunk to the LLM to preserve global context.

---

## 3. Retrieval & Ranking Mechanics

### 3.1 Limitations of Pure Vector Search
Pure dense vector search maps queries and chunks into a continuous embedding space. While effective for semantic concepts, it suffers from:
1. **Out-of-Vocabulary (OOV) & Exact Keyword Failures**: Rare acronyms (e.g. `RS256`, `mTLS`), error logs, and serial numbers do not possess rich semantic neighborhoods.
2. **Hubness Problem**: In high dimensions, certain points ("hubs") tend to appear among the nearest neighbors of disproportionately many query points.

### 3.2 BM25 Okapi (Lexical Sparse Search)
BM25 scores documents based on term frequency ($TF$) and inverted document frequency ($IDF$), penalized by document length:

$$Score(D, Q) = \sum_{q_i \in Q} IDF(q_i) \cdot \frac{TF(q_i, D) \cdot (k_1 + 1)}{TF(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- $k_1 = 1.5$: Governs term frequency saturation.
- $b = 0.75$: Controls length normalization penalty.

### 3.3 Reciprocal Rank Fusion (RRF)
Fusing BM25 scores (unbounded: $[0, \infty)$) and dense cosine similarities ($[-1, 1]$) with linear addition requires fragile min-max normalization. **RRF** overcomes this by utilizing discrete ranks:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k=60$ acts as a constant dampening the advantage of the absolute top ranks, ensuring balanced representation between lexical and semantic matches.

### 3.4 Cross-Encoder Reranker
- **Bi-Encoder**: Encodes $Query$ and $Doc$ independently: $Score = f(Q) \cdot f(D)$. Complexity is $O(1)$ per candidate using pre-indexed vectors.
- **Cross-Encoder**: Passes $[CLS] \circ Query \circ [SEP] \circ Doc \circ [SEP]$ simultaneously into multi-head self-attention layers. Every query token attends to every document token, capturing nuanced syntactic dependencies and negations.
- **Production Paradigm**: Overfetch $K=20$ candidates with Hybrid RRF in $\approx 5\text{ms}$, then rerank down to $N=4$ candidates with Cross-Encoder in $\approx 20\text{ms}$.

---

## 4. Query Transformations

1. **Query Intent Routing**: Distinguishes between chit-chat, direct logic, and domain retrieval to save latency and token costs.
2. **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer document via LLM, then embeds that passage. Shifts the retrieval geometry from *Question $\rightarrow$ Document* to *Document $\rightarrow$ Document*.

---

## 5. Evaluation Framework (RAG Triad)

| Metric | Mathematical / Algorithmic Formulation | Failure Mode Caught |
|---|---|---|
| **Faithfulness** | $\frac{|\text{Supported Claims in Answer}|}{|\text{Total Claims in Answer}|}$ | Hallucinations & fabrications |
| **Answer Relevance** | $\text{CosineSim}(\mathbf{e}_{query}, \mathbf{e}_{answer})$ | Rambling or off-topic responses |
| **Context Precision** | $\frac{\sum_{k=1}^K Precision@k \times v_k}{\text{Total Relevant Candidates}}$ | Low-relevance noise drowning top ranks |
| **Context Recall** | $\frac{|\text{Retrieved Ground Truth Facts}|}{|\text{Total Ground Truth Facts}|}$ | Missing key information |
| **MRR** | $\frac{1}{\text{Rank of first relevant chunk}}$ | Delayed retrieval of core signal |
