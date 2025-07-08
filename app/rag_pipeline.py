# rag_pipeline.py
import os
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import streamlit as st
load_dotenv()

# Load models and resources
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    vector_store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store"))
    index = faiss.read_index(os.path.join(vector_store_dir, "faiss_index.bin"))
    with open(os.path.join(vector_store_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

@st.cache_resource
def load_client():
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

# RAG logic
def retrieve_chunks(query, embed_model, index, metadata, top_k=5):
    query_emb = embed_model.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k)
    results = []
    seen_texts = set()
    for i in I[0]:
        chunk = metadata[i]
        if chunk["chunk_text"] not in seen_texts:
            results.append(chunk)
            seen_texts.add(chunk["chunk_text"])
    return results

def build_prompt(query, chunks):
    context = "\n\n".join([c["chunk_text"] for c in chunks])
    prompt = f"""
    You are an AI assistant built to support internal stakeholders and decision-makers at CrediTrust Financial.
    Your role is to analyze customer complaints and provide clear, concise, and insightful answers to questions raised by company teams.
    Use the following retrieved complaint excerpts as your source of truth.
    If the context does not contain enough information to answer the question, respond by stating that explicitly.
    
    INSTRUCTIONS:
    - Use ONLY the context below.
    - Keep your answer clear, concise, and under 200 words.
    - Organize your response into small paragraphs.
    - Avoid repetition.

    Context:
    {context}

    Question: {query}

    Answer:"""
    return prompt

def generate_answer(prompt, client, max_words=200):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions using customer complaint data."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            max_tokens=512,
            temperature=0.5,
        )

        if not response or not response.choices:
            return "⚠️ No response generated from the model."

        raw_answer = response.choices[0].message.content.strip()

        for delimiter in ["\n\nQuestion:", "\nQuestion:", "Question:"]:
            if delimiter in raw_answer:
                raw_answer = raw_answer.split(delimiter)[0].strip()
                break

        words = raw_answer.split()
        if len(words) > max_words:
            raw_answer = " ".join(words[:max_words]) + "..."

        return raw_answer

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
