# ⚡ Local AI-Powered RAG System Explorer

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-orange.svg)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-green.svg)](https://github.com/langchain-ai/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive, modular playground for exploring and benchmarking **100% local, offline Retrieval-Augmented Generation (RAG) pipelines**. This repository documents hands-on experiments with various chunking strategies, multi-stage retrieval pipelines, reranking, and local LLM evaluation metrics using Ollama, ChromaDB, and FlashRank.

---

## 🏗️ System Architecture

### 1. Ingestion Pipeline
```mermaid
graph TD
    A[Raw Docs: txt / pdf] --> B(Chunking Strategies)
    B -->|Recursive Character| C[Overlap Chunks]
    B -->|Semantic Invariant| D[Embed Distance Chunks]
    B -->|Agentic LLM-Guided| E[LLM Topic Boundary Chunks]
    C & D & E --> F[Local Embeddings: nomic-embed-text]
    F --> G[(Local Vector Store: ChromaDB)]
```

### 2. Retrieval & Generation Pipeline
```mermaid
graph TD
    UserQuery[User Query] --> A(Dense Retriever: ChromaDB)
    UserQuery --> B(Sparse Retriever: BM25)
    A & B --> C[Ensemble Retriever: Hybrid Search]
    C -->|Candidate Pool: k=15| D[Local Reranker: FlashRank ms-marco]
    D -->|Refined Context: top_n=3| E[Local LLM: Llama 3.2 3B]
    E --> Answer[Final Grounded Answer]
```

---

## 🌟 Key Features

*   **100% Offline & Private:** Zero external API keys needed (no OpenAI or Cohere charges). Every step runs on local hardware.
*   **Diverse Chunking Strategies:** Compare standard Recursive Character, Semantic distance, and Agentic (LLM-based) splitting.
*   **Multi-Stage Hybrid Search:** Combine keyword matching (BM25) with vector embeddings (Cosine Similarity) using LangChain's `EnsembleRetriever`.
*   **Local Reranking:** Utilize `FlashRank` (`ms-marco-MiniLM-L-6-v2`) via ONNX on CPU to reorder documents by relevance before context windowing.
*   **Sentence-Window Retrieval:** Index single sentences to achieve precise cosine match vectors, but feed the adjacent sentences (sentence window) to the LLM.
*   **HyDE (Hypothetical Document Embeddings):** Prompt a local model to generate a mock answer first, then embed that mock document to query your index.
*   **Local LLM-as-a-Judge Evaluation:** Run offline quality tests to calculate **Faithfulness (Groundedness)** and **Answer Relevance** scores dynamically.
*   **Interactive Web UI Dashboard:** Streamlit-powered workspace to test all pipeline components visually.

---

## 📂 Repository Experiment Map

| Filename | Type | Description |
| :--- | :--- | :--- |
| [1_ingestion_pipeline.py](1_ingestion_pipeline.py) | Script | Standard document ingestion, recursive chunking, and ChromaDB indexing. |
| [2_retrieval_pipeline.py](2_retrieval_pipeline.py) | Script | Similarity and cosine distance searches with score thresholds. |
| [3_answer_generation.py](3_answer_generation.py) | Script | Basic generation linking context and user query with Llama 3.2. |
| [4_history_aware_generation.py](4_history_aware_generation.py) | Script | Conversational RAG with query reformulation based on chat logs. |
| [5_recursive_character_text_splitter.py](5_recursive_character_text_splitter.py) | Script | Character vs. Recursive Character text splitter comparisons. |
| [6_semantic_chunking.py](6_semantic_chunking.py) | Script | Dynamic splitting at semantic percentile shifts using embeddings. |
| [7_agentic_chunking.py](7_agentic_chunking.py) | Script | Splitting documents by prompting an LLM to add topic split markers. |
| [8_multi_modal_rag.ipynb](8_multi_modal_rag.ipynb) | Notebook | Parsing complex PDFs (extracting tables, text, and images) using a local vision model (`moondream`). |
| [9_retrieval_methods.py](9_retrieval_methods.py) | Script | Comparing Vector Search, Thresholding, and Maximum Marginal Relevance (MMR). |
| [10_multi_query_retrieval.py](10_multi_query_retrieval.py) | Script | Query expansion by translating a search query into multiple variations. |
| [11_reciprocal_rank_fusion.py](11_reciprocal_rank_fusion.py) | Script | Implementing Reciprocal Rank Fusion (RRF) to merge query variations. |
| [12_hybrid_search.ipynb](12_hybrid_search.ipynb) | Notebook | Combining Vector Search (Dense) and BM25 Search (Sparse). |
| [13_reranker.ipynb](13_reranker.ipynb) | Notebook | Local multi-stage retrieval + FlashRank compressor + ChatOllama LLM. |
| [14_hyde_retrieval.py](14_hyde_retrieval.py) | Script | Hypothetical Document Embeddings (HyDE) retrieval pipeline. |
| [15_sentence_window_retrieval.py](15_sentence_window_retrieval.py) | Script | Sentence-level indexing with adjacent context window expansion. |
| [16_rag_evaluation.py](16_rag_evaluation.py) | Script | LLM-as-a-Judge scoring for Faithfulness and Relevance offline. |
| [app.py](app.py) | App | Interactive Streamlit Web Dashboard playground. |

### 📊 Streamlit App Feature Comparison
To help navigate the interactive dashboard, here is a quick summary comparing the **Retrieval Playground (Tab 2)** and the **LLM-as-a-Judge Benchmarks (Tab 3)**:

| Feature | Tab 2: Retrieval Playground | Tab 3: LLM-as-a-Judge Benchmarks |
| :--- | :--- | :--- |
| **Pipeline Step** | Retrieval Only | Full Pipeline (Retrieve → Generate → Evaluate) |
| **Outputs** | Raw source chunks | Conversational answer, latency, and quality scores |
| **LLM Generation** | No LLM-generated answer | Yes (Llama 3.2 writes the final response) |
| **Evaluation** | Visual comparison of ranks | Automated Faithfulness & Relevance metrics |

---

## ⚡ Quick Start Guide

### 1. Install Ollama & Pull Models
Download [Ollama](https://ollama.com) and pull the models required for execution:
```bash
# General LLM (Llama 3.2 3B)
ollama pull llama3.2:3b

# General Embeddings (Nomic Text Embeddings)
ollama pull nomic-embed-text

# Vision Model (Required for Multi-Modal notebook 8)
ollama pull moondream
```

### 2. Set Up Virtual Environment & Dependencies
Initialize a virtual environment and install dependencies listed in `requirements.txt`:
```bash
# Clone the repository
git clone https://github.com/your-username/local-rag-explorer.git
cd local-rag-explorer

# Create & activate virtualenv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 3. Ingest Sample Documents
Set up your vector index by processing the sample company data located in the `docs` directory:
```bash
python 1_ingestion_pipeline.py
```

### 4. Run the Streamlit Dashboard
Launch the interactive browser application to explore your local pipelines visually:
```bash
streamlit run app.py
```

---

## 📈 Performance & VRAM Optimization Tips
*   **GPU Offloading:** Ensure Ollama utilizes your discrete GPU if available. Ollama handles GPU offloading automatically, but you can check resource consumption with `nvidia-smi` (on NVIDIA hardware).
*   **Model Parameter Sizes:** Llama 3.2 3B is highly optimized for consumer hardware. If your workstation has limited memory (e.g. less than 8GB RAM), consider switching to `gemma2:2b` or `phi3:latest`.
*   **Context Window Limitations:** Ensure `num_ctx` is set appropriately in your `ChatOllama` configurations (default is 2048 tokens). RAG systems with long context windows (e.g., retrieving 10+ chunks) may experience performance slowdowns on CPU.