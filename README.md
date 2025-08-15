# 🧠 CrediTrust Complaint Insight Chatbot (RAG-based)

This project is an internal AI tool for **CrediTrust Financial** that helps product managers, support, and compliance teams analyze and extract insights from thousands of unstructured customer complaints across multiple financial services.

Built using **Retrieval-Augmented Generation (RAG)**, the system answers plain-English questions using real customer complaint narratives from the **Consumer Financial Protection Bureau (CFPB)** dataset.

---
-<p align="center">
  <img src="https://github.com/user-attachments/assets/0cf0e328-b228-4e26-8d2d-570276bb3792" 
       alt="image" 
       width="698" 
       height="556" 
       style="border: 5px solid white;">
</p>

---

## 🚀 Project Goals

- 📉 **Reduce time** to detect complaint trends from days to minutes.
- 🧑‍💼 **Empower non-technical teams** to explore complaints using natural language.
- 🔍 **Proactively surface** product or compliance risks across:
  - Credit Cards  
  - Personal Loans  
  - Buy Now, Pay Later (BNPL)  
  - Savings Accounts  
  - Money Transfers

---

## 🛠️ Key Features

- ✅ Vector similarity search via **FAISS**
- ✅ Embedding model using `sentence-transformers/all-MiniLM-L6-v2`
- ✅ Cleaned and preprocessed real-world complaint data
- ✅ Support for **multi-product filtering**
- ✅ Built with `LangChain`, `Transformers`, and `DVC`
- ✅ Ready for **Streamlit/Gradio UI** (optional)

---
## ⚙️ Setup Instructions

1. **Clone the repo:**

```bash
git clone https://https://github.com/Henamen21/Complaint-Analysis-With-RAG.git
cd Complaint-Analysis-With-RAG
```
2. **Create virtual envirnment**
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```
3. Install Depenencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install dvc
```
4. Pull dvc
```bash
dvc pull
```
🧠 Model Info
Embedding model: sentence-transformers/all-MiniLM-L6-v2

Language model: Open to extension (e.g., OpenAI, LLaMA, FLAN-T5, etc.)

Vector DB: FAISS (Local)


