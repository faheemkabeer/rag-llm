# 🤖 RAG Chatbot using Gemini 2.5 + FAISS

A powerful AI assistant that answers questions from any uploaded **PDF document** using **Google Gemini 2.5 Pro** and **FAISS**-based vector search. This chatbot implements **Retrieval-Augmented Generation (RAG)** to ground responses in actual document content, increasing accuracy and reducing hallucination.

---

## 📌 Project Overview

This chatbot combines document retrieval and large language model (LLM) generation in a seamless web interface. Instead of relying only on the LLM’s memory, it intelligently pulls relevant content from an uploaded PDF and uses that as context for Gemini to generate a grounded response.

> Example: Upload a Python PDF and ask “Explain the logic behind prime number checking.” The bot will fetch the relevant section and answer in plain language.

---

## 🧱 Architecture

```text
User → Uploads PDF
         ↓
  PyMuPDF → Extract text
         ↓
Split into overlapping chunks
         ↓
Encode with SentenceTransformer
         ↓
Store embeddings in FAISS
         ↓
User asks a query
         ↓
Embed the query → Search top-k chunks
         ↓
Pass context + question → Gemini 2.5 Pro
         ↓
             ⬇
         Generated answer
⚙️ Technologies Used
Component	Library / Tool
PDF Text Extraction	PyMuPDF
Vector Search	FAISS
Embeddings	sentence-transformers
Language Model	Gemini 2.5 Pro (Google Generative AI)
Web Interface	Streamlit

🧠 Key Features
📄 Upload any .pdf document (lecture notes, guides, etc.)

🔍 Asks questions and retrieves the most relevant sections

💬 Gemini generates natural language answers from context

🧠 Embedding-based retrieval to ensure factual grounding

⚡ Fast, responsive, and runs entirely on your machine

📦 Installation
1. Clone this Repository
bash
Copy
Edit
git clone https://github.com/yourusername/rag-chatbot-gemini.git
cd rag-chatbot-gemini
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔐 API Key Setup
Create a free Gemini API key at https://makersuite.google.com/app/api

Edit app.py and replace:

python
Copy
Edit
genai.configure(api_key="YOUR_API_KEY_HERE")
🔒 Tip: Use environment variables or .env files to avoid hardcoding secrets.

🚀 Run the App
bash
Copy
Edit
streamlit run app.py
Then open http://localhost:8501 in your browser.

📁 File Structure
bash
Copy
Edit
rag-chatbot-gemini/
├── app.py               # Main application logic
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
📌 Use Cases
📚 Study assistant for students — Ask questions from lecture notes

📄 Legal/Finance document summarization

🧠 Quick reference for technical documentation

🗃️ Personal knowledge base with offline PDFs
