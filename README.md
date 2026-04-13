# 📊 Mutual Fund FAQ Assistant (RAG Pipeline)

An enterprise-grade, "facts-only" RAG (Retrieval-Augmented Generation) Chatbot that answers questions exclusively using data scraped from 15 HDFC mutual fund schemes on Groww.

![Streamlit UI Demo](Docs/Images/ui_demo.png) *(Note: Add the UI screenshot to Docs/Images/ui_demo.png if it exists!)*

## 🚀 Key Features

* **Facts Only:** Includes a multi-layer query classifier that serves as an "iron shield". It automatically blocks PII (PAN, Aadhaar, Phone Numbers), refuses to give financial/investment advice, and forces the bot to only stick to factual data.
* **Hybrid Search Retrieval:** Combines Semantic search using local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via ChromaDB and Sparse/Keyword search (BM25) via a local SQLite corpus. Fused seamlessly with Reciprocal Rank Fusion (RRF, k=60).
* **High-Speed Inference:** LLM generation powered by the blazing-fast Groq API (`llama-3.3-70b-versatile`). (Easily swappable with OpenAI).
* **Automated Data Pipeline:** A GitHub Actions workflow runs daily at 9:15 AM IST to scrape live data from Groww UI, ensuring the bot always has the latest NAVs and AUMs.
* **Post-Generation Guardrails:** A strict validation layer ensures the LLM's answers are exactly ≤ 3 sentences, always contain a source citation link, and have zero hallucinated numbers.

---

## 🏗️ Architecture

The project is built in 5 systematic phases:

1. **Ingestion Layer:** Scraping and parsing of Groww Web pages.
2. **Chunking & Embedding:** Smart document parsing (Header, Returns, Fund Manager, Exit Load, etc.) embedded locally into a unified Vector+SQLite store.
3. **Query Pipeline:** Multi-layer intent classification $\rightarrow$ Hybrid Search $\rightarrow$ LLM Generation $\rightarrow$ Guardrail Validation.
4. **UI Setup:** A beautiful dark-glassmorphic themed Streamlit interface with a persistent SQLite-backed native conversation memory (Thread Management).
5. **Evaluation Engine:** Built-in scripts to verify the pipeline metrics against Gold-Standard QA pairs measuring Recall@3, MRR, and Refusal Accuracy.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Frontend UI:** Streamlit
* **Vector DB:** ChromaDB
* **Document Store & Chat Threads:** SQLite3
* **Embeddings:** `all-MiniLM-L6-v2` (Local via HuggingFace)
* **LLM Provider:** Groq (`llama-3.3-70b-versatile`) API (Configurable)
* **Keyword Search:** `rank_bm25`

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yadav1995/rag-chatbot-mutual-funds.git
cd rag-chatbot-mutual-funds
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory (you can use `.env.example` as a template). To use Groq (Recommended, Free API):

```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Run the data pipeline (Optional)
This repository might already contain an embedded Vector DB in `data/` if run previously. If you need to re-fetch and rebuild the database:
```bash
# 1. Scrape all 15 URLs
python src/ingestion/scraping_service.py

# 2. Chunk the documents
python src/ingestion/chunker.py

# 3. Embed the corpus
python src/ingestion/embedder.py --rebuild
```

### 5. Launch the Streamlit App
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Evaluation

You can run the exhaustive 85-test suite to verify the classifiers, guardrails, thread management, and pipeline integrations.
```bash
python -m pytest tests/ -v
```

To run the custom evaluation pipeline (Scores Recall@3, MRR, and Intent class accuracy):
```bash
python tests/eval_pipeline.py
```

---

## ⚠️ Disclaimer
**This project does NOT give out financial advice.** It is built to strictly block advisory questions. It is a technological demonstration of RAG systems. Data in this repository is dynamically retrieved from public URLs.
