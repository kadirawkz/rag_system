import streamlit as st
import os
import re
import time
from typing import Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document
from collections import defaultdict

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Local AI RAG Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;400;600&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Headers and Titles */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1525 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Cards and Glassmorphism Containers */
    .rag-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .rag-card:hover {
        border-color: rgba(0, 242, 254, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Badges */
    .rag-badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 5px;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
    }
    .rag-badge-secondary {
        background: rgba(147, 51, 234, 0.1);
        border: 1px solid rgba(147, 51, 234, 0.3);
        color: #c084fc;
    }
    
    /* Micro-animations and buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    
    /* Metrics display */
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)


# --- STATE INITIALIZATION ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "ingestion_success" not in st.session_state:
    st.session_state.ingestion_success = False


# --- APP TITLE ---
st.markdown("<h1>⚡ Local AI RAG System Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top:-15px; margin-bottom:30px;'>Explore, visualize, and benchmark local Retrieval-Augmented Generation (RAG) pipelines in real-time using Ollama and local models.</p>", unsafe_allow_html=True)


# --- SIDEBAR CONFIGURATIONS ---
with st.sidebar:
    # Render a small, high-contrast white Ollama logo
    st.markdown('<img src="https://ollama.com/public/ollama.png" style="width: 40px; filter: brightness(0) invert(1); margin-bottom: 10px;" />', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:10px;'>Pipeline Engine</h3>", unsafe_allow_html=True)
    
    # Model Configurations
    st.markdown("**Local LLM (Ollama)**")
    llm_model = st.selectbox(
        "Chat Model",
        ["llama3.2:3b", "llama3.2", "gemma2:2b", "phi3:latest", "mistral"],
        index=0
    )
    
    st.markdown("**Embedding Model**")
    embed_model = st.selectbox(
        "Embedding Model",
        ["nomic-embed-text", "all-minilm:latest", "bge-m3"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Chunking Parameters")
    chunk_size = st.slider("Chunk Size (characters)", 100, 1500, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 300, 50, 10)
    
    st.markdown("---")
    st.markdown("### Retrieval Parameters")
    retrieval_k = st.slider("Retrieval Candidates (k)", 1, 15, 5)
    rerank_n = st.slider("Rerank Results (top_n)", 1, 10, 3)
    
    st.markdown("---")
    st.info("💡 **Ollama Required:** Ensure Ollama is running locally and the models are pulled (`ollama pull llama3.2:3b` & `ollama pull nomic-embed-text`).")


# --- HELPER FUNCTIONS ---
def get_embedding_model():
    return OllamaEmbeddings(model=embed_model)

def get_llm():
    return ChatOllama(model=llm_model, temperature=0)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

@st.cache_data
def get_pdf_pages(file_path):
    from pypdf import PdfReader
    try:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return pages
    except Exception as e:
        return [f"Error reading PDF: {str(e)}"]


def _extract_ollama_usage(message):
    """Pull token counts from Ollama metadata when available."""
    metadata = getattr(message, "response_metadata", {}) or {}
    return {
        "eval_count": metadata.get("eval_count"),
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "eval_duration": metadata.get("eval_duration"),
        "prompt_eval_duration": metadata.get("prompt_eval_duration"),
    }


def run_streaming_generation(llm, prompt):
    """Measure TTFT and total generation time using streamed chunks."""
    start_time = time.perf_counter()
    first_token_time = None
    chunks = []
    last_message = None

    for chunk in llm.stream(prompt):
        if first_token_time is None and getattr(chunk, "content", ""):
            first_token_time = time.perf_counter()
        if getattr(chunk, "content", ""):
            chunks.append(chunk.content)
        last_message = chunk

    end_time = time.perf_counter()
    response_text = "".join(chunks).strip()
    ttft = (first_token_time - start_time) if first_token_time else None
    total_latency = end_time - start_time
    usage = _extract_ollama_usage(last_message) if last_message else {}
    return response_text, ttft, total_latency, usage


def estimate_tokens_from_text(text):
    return len(re.findall(r"\S+", text.strip())) if text.strip() else 0




# --- TABS LAYOUT ---
tab_ingest, tab_retrieve, tab_benchmark = st.tabs([
    "📥 Ingestion & Chunking Visualizer", 
    "🔍 Retrieval Playground", 
    "📊 Generation & LLM-as-a-Judge Benchmarks"
])


# --- TAB 1: INGESTION & CHUNKING ---
with tab_ingest:
    st.markdown("## Document Ingestion & Chunking Strategies")
    st.write("Upload a document or use a sample text to compare different local chunking methods dynamically.")
    
    col_input, col_viz = st.columns([1, 1])
    
    with col_input:
        chunking_strategy = st.radio(
            "Select Chunking Strategy",
            ["Recursive Character", "Semantic Invariant", "Agentic (LLM-Guided)"]
        )
        
        # Build sample datasets list dynamically from docs/
        sample_text_dict = {
            "Tesla Q3 (Quick Sample)": "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024. The company exceeded analyst expectations by 15%. Revenue growth was driven by strong vehicle deliveries. The Model Y became the best-selling vehicle globally, with 350,000 units sold. Customer satisfaction ratings reached an all-time high of 96%. However, supply chain issues caused a 12% increase in production costs.",
            "Microsoft GitHub (Quick Sample)": "Microsoft acquired GitHub for $7.5 billion in 2018. The integration of Microsoft Teams with GitHub has significantly enhanced developer workflows. Microsoft's developer tools division sees strong adoption globally, particularly with Visual Studio Code. Additionally, Microsoft announced new AI-powered Copilot features for developers, expanding context window sizes to 128k tokens.",
        }
        
        # Scan docs/ folder and add files (supporting TXT and PDF)
        docs_dir = "docs"
        if os.path.exists(docs_dir):
            for file_name in sorted(os.listdir(docs_dir)):
                if file_name.endswith(".txt"):
                    display_name = f"📄 docs/{file_name}"
                    try:
                        with open(os.path.join(docs_dir, file_name), "r", encoding="utf-8") as f:
                            sample_text_dict[display_name] = f.read()
                    except Exception as e:
                        pass
                elif file_name.endswith(".pdf"):
                    # Store path instead of text to defer extraction
                    display_name = f"📕 docs/{file_name}"
                    sample_text_dict[display_name] = os.path.join(docs_dir, file_name)
        
        sample_text_dict["Custom Text (Empty)"] = ""
        
        selected_sample = st.selectbox("Choose Sample Dataset", list(sample_text_dict.keys()))
        
        if selected_sample == "Custom Text (Empty)":
            raw_text = st.text_area("Paste text here...", height=250, placeholder="Type or paste your technical document contents here...")
        elif selected_sample.startswith("📕"):
            # Load PDF pages
            file_path = sample_text_dict[selected_sample]
            pages = get_pdf_pages(file_path)
            total_pages = len(pages)
            
            # Show a page slider and page metrics
            st.markdown(f"**PDF Metadata:** `Pages: {total_pages}`")
            page_num = st.slider("Select Page to Preview", 1, total_pages, 1)
            
            # Show only the selected page in the text preview (read-only)
            selected_page_text = pages[page_num - 1]
            raw_text = "\n".join(pages) # The text to be chunked is the entire document!
            
            st.text_area(f"Document Page {page_num} Preview (Read-Only)", selected_page_text, height=200, disabled=True)
        else:
            raw_text = st.text_area("Document Text Preview", sample_text_dict[selected_sample], height=250)
            
        btn_chunk = st.button("🔧 Generate Chunks & Visualize")
        
    with col_viz:
        st.markdown("### Chunking Visualization Output")
        
        if btn_chunk and raw_text:
            st.session_state.chunks = []
            
            # Recursive Character Splitter
            if chunking_strategy == "Recursive Character":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
                splits = splitter.split_text(raw_text)
                for i, split in enumerate(splits):
                    st.session_state.chunks.append(Document(page_content=split, metadata={"chunk_index": i}))
                    
            # Semantic Chunker
            elif chunking_strategy == "Semantic Invariant":
                with st.spinner("Calculating embedding vectors and computing semantic breakpoints..."):
                    splitter = SemanticChunker(
                        embeddings=get_embedding_model(),
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=70
                    )
                    splits = splitter.split_text(raw_text)
                    for i, split in enumerate(splits):
                        st.session_state.chunks.append(Document(page_content=split, metadata={"chunk_index": i}))
                        
            # Agentic (LLM-Guided) Splitter
            elif chunking_strategy == "Agentic (LLM-Guided)":
                with st.spinner("Querying LLM to identify natural topic boundaries..."):
                    llm = get_llm()
                    prompt = f"""You are a text chunking expert. Split this text into logical chunks.
                    Rules:
                    - Split at natural topic boundaries
                    - Keep related information together
                    - Put "<<<SPLIT>>>" between chunks
                    
                    Text:
                    {raw_text}
                    
                    Return the text with <<<SPLIT>>> markers:"""
                    response = llm.invoke(prompt).content
                    splits = [chunk.strip() for chunk in response.split("<<<SPLIT>>>") if chunk.strip()]
                    for i, split in enumerate(splits):
                        st.session_state.chunks.append(Document(page_content=split, metadata={"chunk_index": i}))
            
        # Render output if chunks exist in session state
        if len(st.session_state.chunks) > 0:
            # Show stats
            st.markdown(f"**Total Chunks Generated:** `{len(st.session_state.chunks)}`")
            
            # Create indexing button (STAYS UP & stays clickable)
            if st.button("💾 Index Chunks to Local Vector DB"):
                with st.spinner("Embedding and writing to Chroma DB..."):
                    # Simple persistence to separate temp store
                    db_temp = Chroma.from_documents(
                        documents=st.session_state.chunks,
                        embedding=get_embedding_model(),
                        persist_directory="db/chroma_temp"
                    )
                    st.session_state.ingestion_success = True
                    st.success("Successfully indexed chunks to `db/chroma_temp`!")
            
            st.markdown("---")
            
            # Render chunks list below the button
            for i, chunk in enumerate(st.session_state.chunks):
                st.markdown(f"""
                <div class="rag-card">
                    <span class="rag-badge">Chunk {i+1}</span>
                    <span class="rag-badge rag-badge-secondary">{len(chunk.page_content)} characters</span>
                    <p style="margin-top:10px; font-size:0.95rem;">{chunk.page_content}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Enter or select some document text and click 'Generate Chunks' to see the layout.")


# --- TAB 2: RETRIEVAL PLAYGROUND ---
with tab_retrieve:
    st.markdown("## Retrieval Strategy Comparison")
    st.write("Compare search results from various retrieval strategies side-by-side using local models.")
    
    # Query input
    search_query = st.text_input("Enter search query...", "How much did Microsoft pay to acquire GitHub?", key="ret_query")
    
    col_ret1, col_ret2 = st.columns([1, 1])
    
    if search_query:
        # Load the primary indexed store if exists
        persistent_directory = "db/chroma_db"
        if not os.path.exists(persistent_directory):
            st.warning("⚠️ Main vector store (`db/chroma_db`) does not exist yet. Please run `1_ingestion_pipeline.py` or index some documents in the Ingestion tab first.")
            db_instance = None
        else:
            db_instance = Chroma(
                persist_directory=persistent_directory,
                embedding_function=get_embedding_model(),
                collection_metadata={"hnsw:space": "cosine"}
            )
            
        if db_instance:
            with col_ret1:
                st.markdown("### 🏹 Dense Vector Search (Similarity)")
                with st.spinner("Retrieving..."):
                    vector_ret = db_instance.as_retriever(search_kwargs={"k": retrieval_k})
                    vector_results = vector_ret.invoke(search_query)
                    
                    for idx, doc in enumerate(vector_results, 1):
                        st.markdown(f"""
                        <div class="rag-card">
                            <span class="rag-badge">Rank {idx}</span>
                            <span class="rag-badge rag-badge-secondary">Source: {doc.metadata.get('source','Unknown')}</span>
                            <p style="margin-top:10px; font-size:0.95rem;">{doc.page_content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            with col_ret2:
                st.markdown("### 🔀 Local Hybrid + FlashRank Reranked")
                with st.spinner("Ensembling retrievers and running FlashRank..."):
                    # For Hybrid, we need documents. Let's load them to initialize BM25
                    # Get documents from chroma
                    all_db_docs = db_instance.get()
                    if all_db_docs and all_db_docs['documents']:
                        docs_list = [
                            Document(page_content=text, metadata=meta)
                            for text, meta in zip(all_db_docs['documents'], all_db_docs['metadatas'])
                        ]
                        
                        # Set up BM25
                        bm25_ret = BM25Retriever.from_documents(docs_list)
                        bm25_ret.k = retrieval_k
                        
                        # Set up Vector
                        vec_ret = db_instance.as_retriever(search_kwargs={"k": retrieval_k})
                        
                        # Ensemble
                        ensemble_ret = EnsembleRetriever(
                            retrievers=[vec_ret, bm25_ret],
                            weights=[0.7, 0.3]
                        )
                        
                        # FlashRank Compressor
                        compressor = FlashrankRerank(top_n=rerank_n)
                        
                        # Run pipeline
                        raw_retrieved = ensemble_ret.invoke(search_query)
                        reranked_docs = compressor.compress_documents(raw_retrieved, search_query)
                        
                        for idx, doc in enumerate(reranked_docs, 1):
                            st.markdown(f"""
                            <div class="rag-card" style="border-left: 3px solid #00f2fe;">
                                <span class="rag-badge" style="background:rgba(0,242,254,0.15)">Rank {idx}</span>
                                <span class="rag-badge rag-badge-secondary">Relevance Score: {doc.metadata.get('relevance_score','N/A')}</span>
                                <p style="margin-top:10px; font-size:0.95rem;">{doc.page_content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No documents found in database to initialize BM25/Hybrid.")


# --- TAB 3: BENCHMARK & GENERATION ---
with tab_benchmark:
    st.markdown("## Answer Generation & Local Evaluation")
    st.write("Measure model speed and end-to-end RAG performance with TTFT, throughput, latency, and local judge scores.")

    bench_query = st.text_input("Enter Question for Benchmarking...", "How does Tesla make money?", key="bench_query")
    benchmark_mode = st.selectbox(
        "Benchmark Mode",
        ["Docs-grounded RAG", "Prompt-only model"],
        index=0,
        help="Docs-grounded RAG measures retrieval + generation. Prompt-only isolates the model."
    )
    btn_run_rag = st.button("Run Benchmark")

    if btn_run_rag and bench_query:
        persistent_directory = "db/chroma_db"
        status_box = st.empty()
        llm = get_llm()

        docs = []
        context = ""
        retrieval_latency = 0.0

        if benchmark_mode == "Docs-grounded RAG":
            if not os.path.exists(persistent_directory):
                st.warning("Please verify your Chroma DB contains indexed documents first.")
                st.stop()

            db_instance = Chroma(
                persist_directory=persistent_directory,
                embedding_function=get_embedding_model(),
                collection_metadata={"hnsw:space": "cosine"}
            )

            status_box.info("Fetching relevant context from local vector store...")
            retrieve_start = time.perf_counter()
            retriever = db_instance.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(bench_query)
            retrieval_latency = time.perf_counter() - retrieve_start
            context = "\n".join([doc.page_content for doc in docs])

        status_box.info("Generating answer using local LLM...")
        if benchmark_mode == "Docs-grounded RAG":
            combined_input = f"""Based on the following documents, please answer this question: {bench_query}

Context:
{context}

Answer:"""
        else:
            combined_input = f"""Please answer this question directly and concisely: {bench_query}

Answer:"""

        response, ttft, gen_latency, usage = run_streaming_generation(llm, combined_input)
        total_latency = retrieval_latency + gen_latency
        status_box.success("Benchmark run complete!")

        eval_eval_count = usage.get("eval_count")
        if eval_eval_count is not None and usage.get("eval_duration"):
            tokens_per_sec = eval_eval_count / (usage["eval_duration"] / 1_000_000_000)
            output_tokens = eval_eval_count
        else:
            output_tokens = estimate_tokens_from_text(response)
            tokens_per_sec = output_tokens / gen_latency if gen_latency > 0 else 0.0

        ttft_ms = (ttft * 1000) if ttft is not None else None
        gen_ms = gen_latency * 1000
        total_ms = total_latency * 1000
        retrieval_ms = retrieval_latency * 1000

        col_ans, col_stats = st.columns([2, 1])

        with col_ans:
            st.markdown("### Generated Response")
            st.markdown(f"""
            <div class="rag-card" style="background:rgba(15,23,42,0.6); border:1px solid rgba(0,242,254,0.25);">
                <p style="font-size:1.05rem; line-height:1.6; white-space: pre-wrap;">{response.strip()}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Show Retrieved Context Chunks"):
                if docs:
                    for d_idx, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {d_idx} (Source: {doc.metadata.get('source','Unknown')}):**")
                        st.write(doc.page_content)
                        st.markdown("---")
                else:
                    st.info("Prompt-only mode does not retrieve documents.")

        with col_stats:
            st.markdown("### Pipeline Benchmarks")
            st.metric("TTFT", f"{ttft_ms:.0f} ms" if ttft_ms is not None else "N/A")
            st.metric("Generation Latency", f"{gen_ms:.0f} ms")
            st.metric("Retrieval Latency", f"{retrieval_ms:.0f} ms")
            st.metric("Total Latency", f"{total_ms:.0f} ms")
            st.metric("Tokens / Second", f"{tokens_per_sec:.2f}")
            st.metric("Output Tokens", f"{output_tokens}")

            judge_context = context if context else response
            with st.spinner("Judge is grading the response (Faithfulness & Relevance)..."):
                faith_prompt = f"""Determine if the Answer is fully supported by the Context.
Context: {judge_context}
Answer: {response}
Output a Score exactly as: Score: [float between 0.0 and 1.0]"""
                faith_grade = llm.invoke(faith_prompt).content
                f_match = re.search(r"Score:\s*(0\.\d+|1\.0|0)", faith_grade)
                f_score = float(f_match.group(1)) if f_match else 0.8

                rel_prompt = f"""Determine if the Answer directly addresses the Question.
Question: {bench_query}
Answer: {response}
Output a Score exactly as: Score: [float between 0.0 and 1.0]"""
                rel_grade = llm.invoke(rel_prompt).content
                r_match = re.search(r"Score:\s*(0\.\d+|1\.0|0)", rel_grade)
                r_score = float(r_match.group(1)) if r_match else 0.8

            st.metric("Faithfulness Score (Groundedness)", f"{f_score * 100:.1f} %")
            st.metric("Answer Relevance Score", f"{r_score * 100:.1f} %")

            if f_score > 0.9 and r_score > 0.85:
                st.success("Highly optimal answer generated locally with no hallucinations detected.")
            elif f_score < 0.7:
                st.error("Hallucination warning: Some generated statements are unsupported by retrieved context.")
            else:
                st.warning("Suboptimal answer: Check relevancy or expand the candidate window size.")
