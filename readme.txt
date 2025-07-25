# ScientificAnimals 🧠🌱

**ScientificAnimals** is an AI-powered Flask API that retrieves the scientific name of an animal using a Retrieval-Augmented Generation (RAG) pipeline. It combines Wikipedia and arXiv sources with a Hugging Face large language model.

---

## 👨‍💻 Author
Pedro Arruda Pires

*Empowering AI with knowledge from nature.*

---
  
## 🚀 Features

- 🔍 RAG pipeline with Wikipedia and arXiv document loaders
- 🤖 Open-source LLM: `google/flan-t5-base`
- 📚 Embedding with `sentence-transformers/all-MiniLM-L6-v2`
- ⚡ Fast vector search using FAISS
- 🐍 Python Flask API
- 🐳 Dockerized for easy deployment

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ScientificAnimals.git
cd ScientificAnimals
```

### 2. Build Docker container
```bash
docker build -t scientific-animals .
```

### 3. Run the container
```bash
docker run -d -p 5000:5000 scientific-animals
```

---

## 🧪 How to Use

Send a `POST` request to:
```
http://localhost:5000/scientific-name
```

### Body (JSON):
```json
{
  "animal": "jaguar"
}
```

### Response:
```json
{
  "scientific_name": "Panthera onca"
}
```

---

## 📁 File Structure
```
ScientificAnimals/
├── app.py                 # Flask app with API endpoint
├── rag_chain.py          # Retrieval-Augmented Generation logic
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build config
└── README.md             # This file
```

---

## 🤖 Models Used
- **LLM**: `google/flan-t5-base` — instruction-following transformer
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`

---

## 🛠️ Built With
- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face Hub](https://huggingface.co/models)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Flask](https://flask.palletsprojects.com/)



