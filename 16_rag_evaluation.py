import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import time
import re

load_dotenv()

# Predefined test suite based on the company document collection
EVALUATION_QUESTIONS = [
    {
        "question": "How much did Microsoft pay to acquire GitHub?",
        "ground_truth": "Microsoft acquired GitHub for $7.5 billion in 2018."
    },
    {
        "question": "What was NVIDIA's first graphics accelerator called?",
        "ground_truth": "NVIDIA's first graphics accelerator was called NV1, released in 1995."
    },
    {
        "question": "Who succeeded Ze'ev Drori as CEO of Tesla?",
        "ground_truth": "Elon Musk succeeded Ze'ev Drori as CEO of Tesla in October 2008."
    }
]

def get_rag_response(query, db, llm):
    """Retrieve context and generate answer using standard RAG pipeline"""
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    combined_input = f"""Based on the following documents, please answer this question: {query}

Context:
{context}

Answer:"""
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Use only the provided context to answer the question. If you cannot find the answer, state that you do not have enough information."),
        HumanMessage(content=combined_input),
    ]
    
    response = llm.invoke(messages).content
    return response, context

def evaluate_faithfulness(question, context, answer, judge_llm):
    """Evaluate if the answer is grounded in the retrieved context (No hallucinations)"""
    prompt = f"""You are an independent quality auditor. Your job is to determine if the Answer is fully supported by the Context.
    
    Context:
    {context}
    
    Answer:
    {answer}
    
    Is every statement in the Answer directly supported by the facts in the Context? 
    Analyze the statements one by one.
    Then, output a score between 0.0 and 1.0 (where 1.0 means fully faithful with zero hallucinations, and 0.0 means completely made up or unsupported).
    Format your output exactly as:
    Analysis: [brief reasoning]
    Score: [float value between 0.0 and 1.0]"""
    
    response = judge_llm.invoke(prompt).content
    
    # Extract score using regex
    score_match = re.search(r"Score:\s*(0\.\d+|1\.0|0)", response)
    score = float(score_match.group(1)) if score_match else 0.5
    
    return response, score

def evaluate_answer_relevance(question, answer, judge_llm):
    """Evaluate if the answer directly addresses the question"""
    prompt = f"""You are an independent quality auditor. Your job is to determine if the Answer directly addresses the Question.
    
    Question:
    {question}
    
    Answer:
    {answer}
    
    Does the Answer directly address what is asked? Is it helpful and complete, or is it evasive or irrelevant?
    Analyze the response.
    Then, output a score between 0.0 and 1.0 (where 1.0 means perfectly relevant and helpful, and 0.0 means completely irrelevant).
    Format your output exactly as:
    Analysis: [brief reasoning]
    Score: [float value between 0.0 and 1.0]"""
    
    response = judge_llm.invoke(prompt).content
    
    # Extract score using regex
    score_match = re.search(r"Score:\s*(0\.\d+|1\.0|0)", response)
    score = float(score_match.group(1)) if score_match else 0.5
    
    return response, score

def main():
    print("=== Advanced Local RAG: LLM-as-a-Judge Evaluation Pipeline ===\n")
    
    # 1. Setup local components
    persistent_directory = "db/chroma_db"
    if not os.path.exists(persistent_directory):
        print(f"❌ Chroma vector store not found at '{persistent_directory}'. Please run '1_ingestion_pipeline.py' first.")
        return
        
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # RAG LLM
    rag_llm = ChatOllama(model="llama3.2:3b", temperature=0)
    # Judge LLM (requires low temperature for consistent evaluation)
    judge_llm = ChatOllama(model="llama3.2:3b", temperature=0)
    
    print("Running evaluation suite...\n")
    results = []
    
    for idx, item in enumerate(EVALUATION_QUESTIONS, 1):
        question = item["question"]
        print(f"[{idx}/{len(EVALUATION_QUESTIONS)}] Question: '{question}'")
        
        start_time = time.time()
        # Run RAG
        answer, context = get_rag_response(question, db, rag_llm)
        elapsed_time = time.time() - start_time
        
        # Run Judge
        faith_reason, faith_score = evaluate_faithfulness(question, context, answer, judge_llm)
        relevance_reason, relevance_score = evaluate_answer_relevance(question, answer, judge_llm)
        
        print(f"    RAG Response: \"{answer.strip()}\"")
        print(f"    ✅ Faithfulness (Groundedness) Score: {faith_score:.2f}")
        print(f"    🎯 Answer Relevance Score:            {relevance_score:.2f}")
        print(f"    ⏱️ Latency:                           {elapsed_time:.2f}s\n")
        
        results.append({
            "question": question,
            "answer": answer,
            "faithfulness_score": faith_score,
            "relevance_score": relevance_score,
            "latency": elapsed_time
        })
        
    # Print overall summary
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / len(results)
    avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    print("=" * 60)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Questions Evaluated:  {len(results)}")
    print(f"Average Faithfulness Score: {avg_faithfulness:.2f} (Target: >0.90)")
    print(f"Average Answer Relevance:   {avg_relevance:.2f} (Target: >0.85)")
    print(f"Average Pipeline Latency:   {avg_latency:.2f}s")
    print("=" * 60)
    print("💡 LLM-as-a-Judge enables you to grade output quality automatically and ")
    print("   detect regressions when you change chunk size, embedding models, or reranking weights.")

if __name__ == "__main__":
    main()
