from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=== Advanced Local RAG: Hypothetical Document Embeddings (HyDE) ===")
    
    # 1. Setup local models and DB
    persistent_directory = "db/chroma_db"
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.2:3b", temperature=0.7)  # Higher temperature for generation variety
    
    # Load ChromaDB
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # 2. Define the Query
    query = "What was Microsoft's first hardware product release?"
    print(f"\nUser Query: '{query}'")
    
    # 3. Create HyDE Chain (Generate a hypothetical answer)
    hyde_prompt = PromptTemplate.from_template(
        """You are an expert technical writer. Write a short, detailed paragraph answering the question below.
        Do not worry about being perfectly accurate—focus on writing something that sounds like a passage from an encyclopedia or technical document.

        Question: {question}

        Hypothetical Encyclopedia Passage:"""
    )
    
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    
    print("\n🤖 Generating hypothetical document (HyDE)...")
    hypothetical_doc = hyde_chain.invoke({"question": query})
    print("\n--- Generated Hypothetical Document ---")
    print(hypothetical_doc.strip())
    print("-" * 50)
    
    # 4. Perform Retrieval
    # We will retrieve using both the original query and the hypothetical document to compare them
    print("\n🔍 Retrieving documents...")
    
    # A. Standard Retrieval
    print("\n[A] Standard Retrieval (Query Embeddings):")
    standard_retriever = db.as_retriever(search_kwargs={"k": 3})
    standard_docs = standard_retriever.invoke(query)
    for i, doc in enumerate(standard_docs, 1):
        print(f"  {i}. [Source: {doc.metadata.get('source')}] {doc.page_content[:150]}...")
        
    # B. HyDE Retrieval
    print("\n[B] HyDE Retrieval (Hypothetical Document Embeddings):")
    hyde_docs = db.similarity_search(hypothetical_doc, k=3)
    for i, doc in enumerate(hyde_docs, 1):
        print(f"  {i}. [Source: {doc.metadata.get('source')}] {doc.page_content[:150]}...")
        
    # 5. Let's see the benefit
    print("\n💡 Why HyDE works:")
    print("Standard vector search matches the *query's embeddings* against the *document's embeddings*.")
    print("HyDE matches a *generated answer's embeddings* against the *document's embeddings*.")
    print("This often works better because query questions (short, questioning tone) and knowledge documents ")
    print("(long, declarative tone) are located in different parts of the vector space, whereas a ")
    print("hypothetical answer shares a similar declarative tone and vocabulary with the source documents.")

if __name__ == "__main__":
    main()
