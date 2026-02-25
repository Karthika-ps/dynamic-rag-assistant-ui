from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load embeddings
embeddings = OpenAIEmbeddings()

# Load FAISS index
def load_vector_store(store_path):
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)


def retrieve_relevant_chunks(query, store_path, top_k=8, max_distance=0.42):
    vector_store = load_vector_store(store_path)

    results = vector_store.similarity_search_with_score(query, k=top_k)

    filtered_results = []

    for doc, score in results:
        # For L2 distance: lower = better
        if score <= max_distance:
            filtered_results.append((doc, score))

    return filtered_results



if __name__ == "__main__":
    test_query = "What operational risks or system resilience challenges were highlighted in the report?"

    
    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_score(test_query, k=5)

    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} | Score: {score}")
        print(doc.page_content[:300])
        print("\n" + "-"*50 + "\n")
        print(doc.metadata)


