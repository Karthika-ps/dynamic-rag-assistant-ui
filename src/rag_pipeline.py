from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from src.retrieval import retrieve_relevant_chunks

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


def assemble_context(chunks, max_chunks=3):
    # Sort by score (ascending = better for L2)
    sorted_chunks = sorted(chunks, key=lambda x: x[1])
    
    selected = sorted_chunks[:max_chunks]
    
    context_text = ""
    for i, (doc, score) in enumerate(selected):
        context_text += f"\n[Chunk {i+1} | Distance: {score}]\n"
        context_text += doc.page_content
        context_text += "\n\n"
    
    return context_text


def answer_query(query, store_path, mode="Question Answering"):

    question_lower = query.lower()

    summary_keywords = [
        "summary",
        "summarize",
        "overview",
        "gist",
        "brief",
        "explain the document",
        "what is this document about"
    ]

    qa_keywords = [
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "list",
        "identify",
        "explain",
        "describe"
    ]

    # Detect summary intent
    if any(keyword in question_lower for keyword in summary_keywords):
        effective_mode = "Document Summary"
    # Detect QA intent
    elif any(question_lower.startswith(keyword) for keyword in qa_keywords):
        effective_mode = "Question Answering"
    # Otherwise fallback to UI selection
    else:
        effective_mode = mode  # fallback to selected radio mode

    if effective_mode == "Document Summary":
        top_k = 25
        max_distance = 0.6
    else:
        top_k = 8
        max_distance = 0.42
    retrieved = retrieve_relevant_chunks(query, store_path,top_k=top_k,max_distance=max_distance)

    if not retrieved:
        return {
            "answer": "Insufficient information in knowledge base.",
            "sources": []
        }

    context = assemble_context(retrieved)
    if effective_mode  == "Document Summary":
        prompt = f"""
            You are analyzing a technical document.

            Using ONLY the context provided below, generate a structured executive summary.

            Structure your response into these sections:

            1. Overall Purpose of the Document
            2. Key Themes or Topics
            3. Major Findings or Decisions
            4. Important Risks or Strategic Considerations

            Keep it concise, clear, and structured.
            Do not use external knowledge.

            Context:
            {context}
            """
    else:
        prompt = f"""
            You are answering a question based strictly on the provided context.
 
            Answer using ONLY the provided context.
            You may summarize or combine information from multiple chunks.
            If the context does not contain relevant information at all, say: "Insufficient information in knowledge base."
            Do not use external knowledge.


            Context:
            {context}

            Question:
            {query}
            """

    response = llm.invoke(prompt)

    # Prepare source previews
    sources = []
    for doc, score in retrieved[:3]:
        sources.append({
            "distance": float(score),
            "page": doc.metadata.get("page"),
            "preview": doc.page_content[:200]
        })


    return {
        "answer": response.content,
        "sources": sources
    }



if __name__ == "__main__":
    question = "What operational risks were identified in the report?"
    answer = answer_query(question)
    print("\nFinal Answer:\n")
    print(answer)
