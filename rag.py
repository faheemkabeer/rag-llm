import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import streamlit as st
import os
import re

# ----------------------------
# 1. Load PDF and Extract Text
# ----------------------------
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------------------------
# 2. Split Text into Chunks
# ----------------------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------------------------
# 3. Generate Embeddings & Store
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vectorstore(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# ----------------------------
# 4. Retrieve Top Chunks
# ----------------------------
def retrieve(query, index, chunks, k=3):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed), k)
    return [chunks[i] for i in indices[0]]

# ----------------------------
# 5. Query Ollama via REST API
# ----------------------------
def query_ollama(prompt, model="gemma:2b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            result = response.json()["response"]
            return result.strip()
        else:
            return f"❌ Error from Ollama:\n{response.text}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ Failed to connect to Ollama:\n{str(e)}"

# ----------------------------
# 6. RAG Pipeline
# ----------------------------
def rag_chat(query, pdf_path):
    text = load_pdf(pdf_path)
    if not text.strip():
        return "❌ The uploaded PDF appears to have no extractable text."

    chunks = split_text(text)
    index, _, stored_chunks = create_vectorstore(chunks)
    top_contexts = retrieve(query, index, stored_chunks)

    context = "\n\n".join(top_contexts)
    prompt = f"""You are an AI assistant. Use the below context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

    return query_ollama(prompt)

# ----------------------------
# 7. Streamlit Interface
# ----------------------------
st.set_page_config(page_title="RAG Chatbot with Ollama")
st.title("🤖 RAG Chatbot using Ollama + FAISS")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask something from the document:")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if query:
        with st.spinner("🧠 Thinking..."):
            answer = rag_chat(query, "temp.pdf")
        st.markdown("### 📌 Answer")
        st.write(answer)
