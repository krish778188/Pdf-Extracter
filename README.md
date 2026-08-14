# Pdf-Extracter

A fast, lightweight Retrieval-Augmented Generation (RAG) web application built with **Chainlit**, **Mistral AI**, and **Hugging Face**. Upload any PDF document and ask questions to extract precise, document-grounded answers instantly.

---

## 🚀 Key Features

* **Dynamic PDF Uploads:** Upload any PDF directly through a clean, intuitive web interface.
* **⚡ Instant Local Embeddings:** Utilizes the lightweight `all-MiniLM-L6-v2` model locally on your machine/server. This eliminates API rate limits (e.g., HTTP 429 quota exhaustion) and delivers near-zero latency vector generation.
* **Diverse Retrieval with MMR:** Uses **Maximal Marginal Relevance (MMR)** search to fetch diverse, non-redundant context chunks from the document for more complete answers.
* **Strict Fact-Grounding:** Powered by **Mistral AI (`mistral-large-latest`)** with strict system instructions to answer *only* from the provided PDF context and avoid hallucinations.
* **Ephemeral In-Memory Database:** Vector stores are built directly in memory per session and discarded upon session reset—ensuring privacy and zero disk clutter.
* **Optimized Token Efficiency:** Intentionally designed **without conversational chat history** to drastically reduce API token consumption against free-tier limits and maximize response speed for standalone queries.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **User Interface** | [Chainlit](https://chainlit.io/) |
| **LLM (Reasoning)** | [Mistral AI (`mistral-large-latest`)](https://mistral.ai/) via `langchain-mistralai` |
| **Embedding Model** | [Hugging Face `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (In-Memory) |
| **Framework** | LangChain / LangChain Classic |
| **Document Loader** | PyPDF |

---

## 💡 Architecture & Design Choices

### 1. Why Local Embeddings?
Cloud-based embedding APIs (such as free-tier endpoints) enforce strict requests-per-minute (RPM) limits, causing pipeline bottlenecks and `429 RESOURCE_EXHAUSTED` errors during PDF chunk ingestion. Running `all-MiniLM-L6-v2` locally (~90 MB) removes network overhead, bypasses rate limits entirely, and runs seamlessly even on lightweight CPU/RAM environments.

### 2. Why No Conversation History?
Sending accumulated multi-turn chat history on every retrieval query inflates input token payloads exponentially. To keep the agent **100% free-tier friendly**, prevent rapid token exhaustion, and minimize inference latency, the pipeline evaluates each user query independently against the document.

---

## 📦 Installation & Local Setup

### Prerequisites
* Python 3.10+
* A Mistral AI API key (Get one at [console.mistral.ai](https://console.mistral.ai/))

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
