# Local RAG System

This repository contains a local Retrieval-Augmented Generation (RAG) playground built with Streamlit, LangChain, Chroma, Ollama, and FlashRank.

It is designed to help you experiment with:

- document ingestion and chunking
- dense, sparse, and hybrid retrieval
- reranking with FlashRank
- answer generation and lightweight evaluation

## Project Layout

- `app.py` - Streamlit app for interactive RAG exploration
- `1_ingestion_pipeline.py` - document ingestion into Chroma
- `2_retrieval_pipeline.py` - retrieval-only workflow
- `3_answer_generation.py` - answer generation pipeline
- `4_history_aware_generation.py` - chat history aware generation
- `5_recursive_character_text_splitter.py` - recursive chunking example
- `6_semantic_chunking.py` - semantic chunking example
- `7_agentic_chunking.py` - LLM-guided chunking example
- `9_retrieval_methods.py` - retrieval strategy comparison
- `10_multi_query_retrieval.py` - multi-query retrieval example
- `11_reciprocal_rank_fusion.py` - RRF example
- `14_hyde_retrieval.py` - HyDE retrieval example
- `15_sentence_window_retrieval.py` - sentence window retrieval example
- `16_rag_evaluation.py` - local LLM-as-a-judge evaluation script
- `docs/` - sample source documents used by the demos

## Requirements

- Python 3.10+
- Ollama running locally
- A local Chroma vector store created from your documents

## Install

```bash
pip install -r requirements.txt
```

## Ollama Models

The app expects local Ollama models for both chat and embeddings. A good starting point is:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

You can also try the other models exposed in the Streamlit sidebar, depending on what you have pulled locally.

## Run the Streamlit App

```bash
streamlit run app.py
```

## Evaluate a RAG Run

```bash
python 16_rag_evaluation.py
```

## Data and Generated Files

The repo keeps sample documents in `docs/`, but generated outputs such as Chroma databases, benchmark results, and temporary caches should remain untracked.

If you add new documents for experimentation, place them in `docs/` and keep any large generated artifacts out of version control.
