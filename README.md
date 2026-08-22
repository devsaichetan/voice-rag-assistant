# 🎙️ Voice RAG Assistant

A voice-enabled Retrieval-Augmented Generation (RAG) assistant that allows users to ask questions through **voice or text** and receive grounded answers based only on the retrieved knowledge base.

The system combines **FAISS retrieval, sentence embeddings, cross-encoder reranking, Groq LLM generation, Sarvam Speech-to-Text, and Sarvam Text-to-Speech** into a complete end-to-end voice question-answering pipeline.

---

## 🚀 Features

- 🎙️ Voice-based question answering
- ⌨️ Text-based question answering
- 🗣️ Speech-to-Text using Sarvam AI
- 🔎 Semantic retrieval using FAISS
- 🎯 Cross-encoder reranking
- 🧠 Grounded answer generation using Groq
- 🔊 Text-to-Speech using Sarvam AI
- 📚 Source URL tracking
- ⚡ Retrieval, reranking, generation, STT and TTS latency tracking
- 🌐 FastAPI backend
- 💻 Interactive web frontend
- 🔐 Environment-based API key management
- 🛡️ Similarity threshold filtering to reduce irrelevant answers
- 🎧 Unique generated audio output for each voice request

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      User Input      │
                    │    Voice / Text      │
                    └──────────┬───────────┘
                               │
                     ┌─────────▼─────────┐
                     │    FastAPI API    │
                     └─────────┬─────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Speech-to-Text    │
                    │      Sarvam AI      │
                    └──────────┬──────────┘
                               │
                               ▼
                         User Question
                               │
                    ┌──────────▼──────────┐
                    │ Sentence Embedding  │
                    │  all-MiniLM-L6-v2   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │       FAISS         │
                    │     Retrieval       │
                    └──────────┬──────────┘
                               │
                         Top-K Results
                               │
                    ┌──────────▼──────────┐
                    │  Cross-Encoder      │
                    │     Reranker        │
                    └──────────┬──────────┘
                               │
                         Top-N Results
                               │
                    ┌──────────▼──────────┐
                    │ Similarity Filtering│
                    └──────────┬──────────┘
                               │
                            Context
                               │
                    ┌──────────▼──────────┐
                    │      Groq LLM       │
                    │   GPT-OSS 120B      │
                    └──────────┬──────────┘
                               │
                            Answer
                               │
                    ┌──────────▼──────────┐
                    │     Sarvam TTS      │
                    └──────────┬──────────┘
                               │
                         🔊 Voice Answer
```

---

## 🧠 RAG Pipeline

The system follows a multi-stage Retrieval-Augmented Generation pipeline.

### 1. User Query

The user provides a question through either:

- Voice
- Text

For voice input, the audio is first converted into text using Sarvam Speech-to-Text.

### 2. Embedding

The question is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

### 3. FAISS Retrieval

The generated embedding is searched against the FAISS vector index.

Current index:

```text
Vectors: 8264
```

The system initially retrieves:

```text
TOP_K = 20
```

candidate chunks.

### 4. Cross-Encoder Reranking

The retrieved chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The system keeps:

```text
RERANK_TOP_N = 5
```

top-ranked results.

### 5. Similarity Filtering

Retrieved results are filtered using:

```text
SIMILARITY_THRESHOLD = 0.65
```

If no relevant context remains, the system returns:

```text
I don't have enough information in the provided context.
```

This prevents the LLM from generating unsupported answers.

### 6. Grounded LLM Generation

The filtered context is passed to Groq using:

```text
Model: openai/gpt-oss-120b
```

The model is instructed to answer only using the retrieved context.

### 7. Text-to-Speech

For voice requests, the generated answer is converted back into speech using Sarvam AI.

Each voice request generates a unique audio filename.

---

## 📊 Retrieval Evaluation

The retrieval pipeline was evaluated using:

```text
100 queries
```

### FAISS Only

| Metric | Result |
|---|---:|
| Recall@5 | 0.8500 |
| Recall@10 | 0.9600 |
| MRR | 0.5891 |
| Average Retrieval Latency | ~11.84 ms |

### FAISS + Reranker

| Metric | Result |
|---|---:|
| Recall@5 | 0.9100 |
| Recall@10 | 0.9100 |
| MRR | 0.6225 |
| Average Reranking Latency | ~486.78 ms |
| Average Combined Latency | ~498.62 ms |

### Improvement

```text
Recall@5 improvement:  +0.0600
Recall@10 improvement: -0.0500
MRR improvement:       +0.0334
```

The reranker improved Recall@5 and MRR while adding additional processing latency.

---

## ⚡ Example Runtime Performance

A successful RAG query using Groq produced approximately:

```text
Retrieval:    ~280 ms
Reranking:    ~600 ms
Generation:   ~1.3 seconds
Total:        ~2.2 seconds
```

Actual latency varies depending on the machine, network, API response time, and workload.

---

## 🎤 Voice Pipeline

The complete voice workflow is:

```text
🎙️ User speaks
      ↓
🗣️ Sarvam Speech-to-Text
      ↓
📝 Transcript
      ↓
🔎 FAISS Retrieval
      ↓
🎯 Cross-Encoder Reranking
      ↓
📚 Context Filtering
      ↓
⚡ Groq LLM
      ↓
📝 Generated Answer
      ↓
🔊 Sarvam Text-to-Speech
      ↓
🎧 Audio Response
```

---

## 🛡️ Grounded Answering

The assistant is designed to minimize hallucinations by restricting the LLM to the retrieved context.

The generation prompt follows these principles:

```text
Answer ONLY using the provided context.

Do not use outside knowledge.

Do not invent facts.

If the context does not contain enough information,
say:

"I don't have enough information in the provided context."
```

The system also applies a similarity threshold before sending context to the LLM.

---

## 🧰 Technologies Used

### Backend

- Python
- FastAPI
- Uvicorn

### Retrieval

- FAISS
- Sentence Transformers
- all-MiniLM-L6-v2

### Reranking

- Cross Encoder
- cross-encoder/ms-marco-MiniLM-L-6-v2

### LLM

- Groq
- openai/gpt-oss-120b

### Speech

- Sarvam AI Speech-to-Text
- Sarvam AI Text-to-Speech

### Frontend

- HTML5
- CSS3
- JavaScript
- Browser MediaRecorder API

---

## 📁 Project Structure

```text
Voice_RAG/
│
├── data/
│   ├── processed/
│   │   ├── chunks.json
│   │   └── documents.json
│   │
│   └── raw/
│
├── evaluation/
│
├── indexes/
│   ├── faiss.index
│   └── metadata.json
│
├── src/
│   ├── api.py
│   ├── config.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── rag_engine.py
│   ├── stt.py
│   ├── tts.py
│   └── voice_rag.py
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── tests/
│
├── .gitignore
├── Procfile
├── requirements.txt
└── README.md
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

SARVAM_API_KEY=your_sarvam_api_key
```

Never commit API keys to GitHub.

The `.env` file is excluded using `.gitignore`.

---

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Voice_RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
SARVAM_API_KEY=your_sarvam_api_key
```

### 6. Start the application

```bash
python -m uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

The interactive frontend will be available at:

```text
http://127.0.0.1:8000/app
```

---

## 🔌 API Endpoints

### Health Check

```http
GET /
```

Example response:

```json
{
  "message": "Voice RAG API is running",
  "status": "ok"
}
```

---

### Text Question

```http
POST /ask
```

Request:

```json
{
  "question": "What is the Eiffel Tower?"
}
```

The response contains:

- Question
- Grounded answer
- Retrieved sources
- Retrieval latency
- Reranking latency
- Generation latency
- Total latency

---

### Voice Question

```http
POST /voice-ask
```

Accepts:

```text
.wav
.mp3
.ogg
.m4a
.webm
.flac
```

The endpoint performs:

```text
Audio
  ↓
Speech-to-Text
  ↓
FAISS Retrieval
  ↓
Reranking
  ↓
Groq Generation
  ↓
Text-to-Speech
```

The response includes:

- Original filename
- Transcript
- Answer
- Sources
- Generated audio URL
- STT latency
- Retrieval latency
- Reranking latency
- Generation latency
- TTS latency
- Total latency

---

### Voice Answer

```http
GET /voice-answer/{filename}
```

Returns the generated WAV audio response.

Each voice request receives a unique audio filename.

---

### Frontend

```http
GET /app
```

Opens the interactive Voice RAG web application.

---

## 🌐 Frontend Workflow

The frontend provides:

- 🎙️ Browser microphone recording
- ⏱️ Recording timer
- 🌊 Recording visualization
- 📝 Live transcription/result display
- 💬 Text question input
- 🔎 RAG pipeline visualization
- 📊 Latency information
- 📚 Source display
- 🔊 Generated audio playback
- 📱 Responsive interface

The browser uses the MediaRecorder API to capture voice input and sends the recorded audio to the FastAPI `/voice-ask` endpoint.

---

## 📈 RAG Configuration

The main retrieval parameters are configured in `src/config.py`.

```python
TOP_K = 20

RERANK_TOP_N = 5

SIMILARITY_THRESHOLD = 0.65

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

These values control:

- Number of FAISS candidates
- Number of reranked results
- Minimum similarity required
- Embedding model

---

## 🔐 Security Considerations

The application follows several basic security practices:

- API keys are stored in environment variables.
- `.env` is excluded from Git.
- Virtual environments are excluded from Git.
- Python cache files are excluded.
- Uploaded audio files are temporarily stored and removed after processing.
- Generated audio filenames are unique.
- Voice-answer file requests validate filenames to prevent path traversal.

For production deployment, additional authentication, rate limiting, persistent storage, and monitoring should be considered.

---

## 🚀 Deployment

The project includes a `Procfile`:

```text
web: uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

The deployment platform should provide the following environment variables:

```text
GROQ_API_KEY
GROQ_MODEL
SARVAM_API_KEY
```

API keys should be configured through the deployment platform's environment-variable settings rather than committed to the repository.

---

## 🔮 Future Improvements

- Streaming LLM responses
- Streaming voice responses
- Conversation memory
- Multi-language voice interaction
- Improved handling of speech recognition spelling variations
- Query rewriting
- Hybrid retrieval
- Better reranking optimization
- Persistent cloud audio storage
- User authentication
- Rate limiting
- Request monitoring
- Production logging
- Cloud-based vector database
- Advanced RAG evaluation
- Multi-document knowledge bases

---

## 🎯 Example Interaction

### User

```text
🎙️ What is the Eiffel Tower?
```

### Pipeline

```text
Speech
  ↓
Sarvam STT
  ↓
"What is the Eiffel Tower?"
  ↓
FAISS
  ↓
Cross-Encoder Reranker
  ↓
Relevant Context
  ↓
Groq
  ↓
Grounded Answer
  ↓
Sarvam TTS
  ↓
🔊 Voice Response
```

### Example Answer

```text
The Eiffel Tower is an iron lattice tower located in Paris,
France, completed in 1889. It is a world-famous landmark
that serves as a tourist attraction, an observation point
over the city, and a site for TV and radio broadcast
antennae.
```

---

## 📌 Project Highlights

This project demonstrates an end-to-end implementation of:

```text
Retrieval-Augmented Generation
        +
Semantic Search
        +
Cross-Encoder Reranking
        +
Large Language Model
        +
Speech Recognition
        +
Text-to-Speech
        +
FastAPI
        +
Interactive Web Application
```

The system was evaluated on 100 retrieval queries and achieved:

```text
Recall@5  = 0.91
Recall@10 = 0.91
MRR       = 0.6225
```

with the FAISS + reranker configuration.

---

## 👨‍💻 Author

**Chetan**

Computer Science & Engineering

---

## ⭐ Project Summary

**Voice RAG Assistant** is an end-to-end voice-enabled Retrieval-Augmented Generation application that combines semantic search, FAISS retrieval, cross-encoder reranking, grounded LLM generation, speech recognition, and speech synthesis into a single interactive system.

The project demonstrates how modern AI components can be integrated into a practical real-time voice question-answering application.