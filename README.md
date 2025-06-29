# 🤖 RAG Chatbot using Gemini + FAISS + Streamlit

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** that allows users to upload a PDF document and ask questions about its contents. It uses Google **Gemini Pro** for generating answers, **FAISS** for vector-based semantic search, **Sentence Transformers** for embedding generation, and **Streamlit** for the web interface.

---

## 📚 Table of Contents

1. [🔍 Overview](#-overview)  
2. [🚀 Features](#-features)  
3. [🛠️ Setup Instructions](#️-setup-instructions)  
4. [▶️ Run the App](#️-run-the-app)  
5. [🧾 Example Use Case](#-example-use-case)  
6. [📦 Requirements](#-requirements)  



---

## 🔍 Overview

The app takes a PDF file, extracts its content, splits it into overlapping chunks, converts each chunk into vector embeddings using `SentenceTransformer`, indexes the embeddings using **FAISS**, retrieves the most relevant chunks for a user query, and finally queries **Gemini Pro** to generate a contextual answer.

---

## 🚀 Features

- 📄 PDF Upload
- 🧠 Text Chunking with Overlap
- 🔍 Semantic Search with FAISS
- 🤖 Query Generation using Gemini
- 💻 Easy-to-use Web Interface via Streamlit

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot-gemini.git
cd rag-chatbot-gemini
2. Create and Activate a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set Your Gemini API Key
In app.py, replace:

python
Copy
Edit
genai.configure(api_key="YOUR_API_KEY")
with your actual Gemini API key.

Note: For security, it's better to store API keys in environment variables in production.

▶️ Run the App
bash
Copy
Edit
streamlit run app.py
The app will open in your browser. You can now upload a PDF and ask questions based on its content.

🧾 Example Use Case
Upload a research paper or document, then ask questions like:

“Summarize the conclusion section.”

“What methodology was used?”

“What are the key findings?”

Gemini will respond using only the most relevant context extracted from your document.

📦 Requirements
Here’s what you’ll need (also saved in requirements.txt):

streamlit
PyMuPDF
faiss-cpu
sentence-transformers
numpy
google-generativeai
