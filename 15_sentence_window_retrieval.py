import re
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def split_into_sentences(text):
    """Split text into sentences using simple regex"""
    # Split by period/exclamation/question mark followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def main():
    print("=== Advanced Local RAG: Sentence-Window Retrieval ===")
    
    # 1. Sample text demonstrating the need for context
    sample_text = """
    Artificial Intelligence is transforming industries. 
    In 2024, OpenAI released a new model class. 
    It is called GPT-4o. 
    This model integrates text, vision, and audio natively. 
    Many developers are building voice assistants using it. 
    However, running such models in production requires significant optimization. 
    Quantization is a common technique to reduce model size. 
    It converts 16-bit floats to 8-bit or 4-bit integers.
    """
    
    print("\nOriginal Text:")
    print(sample_text.strip())
    print("-" * 50)
    
    # 2. Process text into sentence windows
    sentences = split_into_sentences(sample_text)
    print(f"\nSplit text into {len(sentences)} individual sentences.")
    
    documents = []
    window_size = 2  # Number of sentences before/after to include in the window
    
    for i, sentence in enumerate(sentences):
        # Calculate start and end indices for the window
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        
        # Combine sentences in the window
        window_context = " ".join(sentences[start:end])
        
        # Create a document containing the single sentence (for vector search)
        # and attach the expanded window context in the metadata
        doc = Document(
            page_content=sentence,
            metadata={
                "sentence_index": i,
                "window_context": window_context,
                "source": "sample_text"
            }
        )
        documents.append(doc)
    
    # 3. Store in an in-memory or temporary local ChromaDB
    print("\nCreating local vector database for sentence-level search...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    persist_dir = "db/chroma_sentence_window"
    
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    # 4. Search and Retrieval
    query = "What is GPT-4o?"
    print(f"\nUser Query: '{query}'")
    
    # Retrieve the single most matching sentence
    results = db.similarity_search(query, k=1)
    
    if results:
        matching_doc = results[0]
        matched_sentence = matching_doc.page_content
        expanded_context = matching_doc.metadata["window_context"]
        
        print("\n--- Search Results ---")
        print(f"🎯 Closest Matching Sentence: \n   \"{matched_sentence}\"")
        print(f"\n🖼️ Expanded Sentence Window Context (adjacent sentences): \n   \"{expanded_context}\"")
        print("-" * 50)
        
        # 5. Generation
        print("\n🤖 Asking LLM to answer using different contexts...")
        llm = ChatOllama(model="llama3.2:3b", temperature=0)
        
        # A. Answer with ONLY the matching sentence
        prompt_limited = f"""Answer the question using ONLY the provided context. If context doesn't answer it, say you don't know.
        Context: {matched_sentence}
        Question: {query}
        Answer:"""
        ans_limited = llm.invoke(prompt_limited).content
        
        # B. Answer with the EXPANDED window context
        prompt_expanded = f"""Answer the question using ONLY the provided context. If context doesn't answer it, say you don't know.
        Context: {expanded_context}
        Question: {query}
        Answer:"""
        ans_expanded = llm.invoke(prompt_expanded).content
        
        print("\n[A] Response with Single Sentence Context:")
        print(f"👉 {ans_limited.strip()}")
        
        print("\n[B] Response with Sentence-Window Context:")
        print(f"👉 {ans_expanded.strip()}")
        
        print("\n💡 Why Sentence-Window Retrieval is powerful:")
        print("1. **Retrieval Precision**: Embedding and searching small sentences prevents dilution of information.")
        print("2. **Generation Context**: Passing the surrounding window provides the LLM with the context it needs ")
        print("   to resolve pronouns (e.g. 'It is called GPT-4o' requires the previous sentence to know what 'It' is) ")
        print("   and construct a complete response.")

    # Cleanup temporary DB files
    # Note: In a real system, you would keep the db, but we can leave it for the user to inspect.

if __name__ == "__main__":
    main()
