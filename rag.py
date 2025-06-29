import fitz  
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai
import os

# ----------------------------
# 1. Configure Gemini API
# ----------------------------
genai.configure(api_key="YOUR_API_KEY_HERE")

# ----------------------------
# 2. Load PDF and Extract Text
# ----------------------------
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------------------------
# 3. Split Text into Chunks
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
# 4. Load Embedding Model (with caching)
# ----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def create_vectorstore(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# ----------------------------
# 5. Retrieve Top Chunks
# ----------------------------
def retrieve(query, index, chunks, k=3):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed), k)
    return [chunks[i] for i in indices[0]]

# ----------------------------
# 6. Query Gemini (now using gemini-2.5-pro)
# ----------------------------
def query_gemini(prompt, model_name="gemini-2.5-pro"):
    try:
        model = genai.GenerativeModel(f"models/{model_name}")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

# ----------------------------
# 7. RAG Pipeline
# ----------------------------
def rag_chat(query, pdf_path):
    text = load_pdf(pdf_path)
    chunks = split_text(text)
    index, _, stored_chunks = create_vectorstore(chunks)
    top_contexts = retrieve(query, index, stored_chunks)

    context = "\n\n".join(top_contexts)
    prompt = f"""You are a helpful and knowledgeable assistant. Use the following context to answer the user's question clearly and concisely.

Context:
{context}

Question: {query}

Answer:"""

    return query_gemini(prompt)

# ----------------------------
# 8. Streamlit Interface
# ----------------------------
st.set_page_config(page_title="RAG Chatbot with Gemini")
st.title("🤖 RAG Chatbot using Gemini + FAISS")

uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
query = st.text_input("❓ Ask something from the document:")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if query.strip():
        with st.spinner("🔍 Analyzing..."):
            answer = rag_chat(query, "temp.pdf")
        st.markdown("### ✅ Answer")
        st.write(answer)

        # ✅ Clean up the uploaded file
        os.remove("temp.pdf")
