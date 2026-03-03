# SHL Assessment Recommendation System

An AI-powered semantic recommendation system that suggests the most relevant **SHL assessments** based on hiring requirements using **LLMs + Vector Search (FAISS)**.

This project builds an intelligent pipeline that understands recruiter queries, extracts hiring intent, and recommends suitable assessments from the SHL catalog.

---

## 🚀 Project Overview

Hiring teams often describe requirements in natural language:

> *"Looking for Python developers with teamwork skills."*

Instead of manually searching through assessment catalogs, this system:

✅ Understands hiring intent using Gemini LLM
✅ Performs semantic similarity search using embeddings
✅ Retrieves best matching SHL assessments
✅ Generates evaluation-ready predictions

---

## 🧠 Architecture

```
User Query
     ↓
Intent Extraction (Gemini)
     ↓
Query Enhancement
     ↓
Embedding Model
     ↓
FAISS Vector Search
     ↓
Top-K Assessment Recommendations
```

---

## ⚙️ Tech Stack

| Component       | Technology           |
| --------------- | -------------------- |
| Backend API     | FastAPI              |
| Frontend        | Streamlit            |
| LLM             | Google Gemini API    |
| Embeddings      | SentenceTransformers |
| Vector Database | FAISS                |
| Language        | Python 3.10+         |
| Logging         | Loguru               |
| Data Handling   | Pandas               |

---

## 📁 Project Structure

```
shl-assessment-recommender/
│
├── app/
│   ├── api/                # FastAPI endpoints
│   ├── core/               # Config & logging
│   ├── embeddings/         # Embedding pipeline
│   ├── evaluation/         # Prediction generation
│   ├── frontend/           # Streamlit UI
│   ├── ingestion/          # Dataset builder
│   ├── llm/                # Gemini intent processing
│   ├── services/           # Recommendation pipeline
│   └── vectorstore/        # FAISS search
│
├── data/
│   ├── source/             # Provided SHL datasets
│   ├── raw/                # JSON catalog
│   ├── processed/          # Clean datasets
│   └── embeddings/         # FAISS index files
│
├── logs/
├── requirements.txt
├── run.bat
└── README.md
```

---

## 🔧 Installation

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd shl-assessment-recommender
```

---

### 2️⃣ Create Environment

```bash
conda create -n shl python=3.10
conda activate shl
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables

Create `.env` file:

```
APP_NAME=SHL-Recommender
API_VERSION=v1

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-flash-latest

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RAW_DATA_PATH=data/raw
PROCESSED_DATA_PATH=data/processed
VECTOR_DB_PATH=data/embeddings

TOP_K_RESULTS=10
```

---

## 📊 Data Pipeline

### Step 1 — Build Dataset

```bash
python -m app.ingestion.build_dataset
```

Creates:

```
data/raw/shl_catalog_raw.json
```

---

### Step 2 — Generate Embeddings + FAISS Index

```bash
python -m app.embeddings.embedder
```

Outputs:

```
data/embeddings/
 ├── embeddings.npy
 ├── faiss_index
 └── metadata.pkl
```

---

## 🔎 Run Recommendation Service

```bash
python -m app.services.recommendation_service
```

Example:

```
Looking for Python developers with teamwork skills
```

Returns recommended SHL assessments.

---

## 🖥️ Run Full Application

### Windows:
```bash
run.bat
```

This starts:
- FastAPI server on `http://127.0.0.1:8000`
- Streamlit UI on `http://127.0.0.1:8501`

### Or manually:

**Terminal 1 - Start API:**
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start UI:**
```bash
streamlit run app/frontend/ui.py
```

---

## 📈 Generate Evaluation Predictions

The SHL test dataset contains hiring requirements. This system predicts the best assessment URL for each.

Run:

```bash
python -m app.evaluation.generate_predictions
```

**Input:** `data/source/test.csv` (with `Query` column)

**Output:** `data/processed/shl_predictions.csv`

Format:

| Query              | Assessment_url       |
| ------------------ | -------------------- |
| Hiring requirement | Recommended SHL link |

✅ Submission-ready CSV.

---

## 🧩 Recommendation Pipeline

### 1. Intent Extraction (Gemini)

Extracts:
* technical skills
* soft skills
* role type
* seniority level

### 2. Query Enhancement

Adds extracted skills to improve retrieval.

### 3. Semantic Retrieval

* Text embeddings generated using SentenceTransformers
* FAISS similarity search over assessment embeddings
* Top-K relevant assessments retrieved

### 4. Optional Explanation

LLM generates reasoning for recommendations.

---

## ⚡ API Endpoints

### Health Check
```
GET /
Response: {"status": "API running"}
```

### Get Recommendations
```
POST /recommend
Request: {"query": "hiring requirement"}
Response: {
  "query": "...",
  "detected_requirements": {...},
  "recommended_assessments": [...],
  "ai_explanation": "...",
  "total_found": 10
}
```

---

## 🧪 Example Query

**Input:**
```
Looking to hire mid-level Python developers with collaboration skills
```

**Output:**
```json
{
  "detected_requirements": {
    "technical_skills": ["Python"],
    "soft_skills": ["Collaboration", "Teamwork"],
    "role_type": "Developer",
    "seniority_level": "Mid-level"
  },
  "recommended_assessments": [
    {
      "name": "Python (New)",
      "description": "Multi-choice test that measures knowledge...",
      "duration": 10,
      "test_type": "K",
      "remote_support": true,
      "url": "https://..."
    },
    ...
  ],
  "ai_explanation": "• Python (New) validates technical skills...\n• Team Dynamics Assessment evaluates collaboration...",
  "total_found": 10
}
```

---

## 📊 Features

* 🧠 LLM-powered intent understanding (Gemini)
* 🔍 Semantic search using FAISS vector database
* ⚡ Fast embedding-based retrieval
* 📱 Interactive Streamlit web interface
* 🚀 FastAPI REST endpoints
* 📊 Batch prediction generation
* 📝 Comprehensive logging
* 🎯 Modular architecture

---

## 🔮 Future Improvements

* Cross-encoder reranking for better relevance
* Skill ontology mapping
* Hybrid BM25 + Vector Search
* User feedback learning loop
* Multi-agent recommendation pipeline
* Caching for frequent queries

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Built as part of SHL Assessment Recommendation challenge using modern LLM + Retrieval architecture.

Combines:
- **Google Gemini** for NLP understanding
- **SentenceTransformers** for semantic embeddings
- **FAISS** for fast vector similarity search
- **FastAPI + Streamlit** for user interfaces
