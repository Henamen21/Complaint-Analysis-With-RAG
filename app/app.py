# app.py
import os
import faiss
import pickle
import streamlit as st
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize HF InferenceClient
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

@st.cache_resource
def load_client():
    return InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

client = load_client()

print('HuggingFace Inference Client initialized.')

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

print('Sentence Transformer model loaded.')

# Safe path to vector store directory
vector_store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store"))
print(f"Vector store directory: {vector_store_dir}")

@st.cache_resource
def load_vector_store():
    index = faiss.read_index(os.path.join(vector_store_dir, "faiss_index.bin"))
    with open(os.path.join(vector_store_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_vector_store()

print('FAISS index and metadata loaded.')

# RAG functions

def retrieve_chunks(query, top_k=5):
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
    Your role is to analyze customer complaints and provide clear, concise, and insightful answers to questions raised by company teams (such as Product, Support, Compliance, and Executives).
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

def generate_answer(prompt, max_words=200):
    # Structure the prompt into a conversational format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions using customer complaint data. "
                "Provide a concise answer and stop once you’ve answered—do not invent new questions."
            )
        },
        {"role": "user", "content": prompt}
    ]

    try:
        # Make a call to the chat-completion endpoint
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",  # or your preferred model
            messages=messages,
            max_tokens=512,
            temperature=0.5,
        )

        # Validate and extract the output
        if not response or not response.choices:
            return "⚠️ No response generated from the model."

        raw_answer = response.choices[0].message.content.strip()

        # Cut off at a new question if appended
        for delimiter in ["\n\nQuestion:", "\nQuestion:", "Question:"]:
            if delimiter in raw_answer:
                raw_answer = raw_answer.split(delimiter)[0].strip()
                break

        # Truncate to 200 words
        words = raw_answer.split()
        if len(words) > max_words:
            raw_answer = " ".join(words[:max_words]) + "..."

        return raw_answer

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


# --- Streamlit Chat Interface ---
st.set_page_config(page_title="CrediTrust Complaint Chat")
st.title("📊 CrediTrust Complaint RAG Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input box at the bottom
if user_query := st.chat_input("Ask a question about customer complaints..."):
    # Append user's question to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        # RAG pipeline
        retrieved = retrieve_chunks(user_query)
        prompt = build_prompt(user_query, retrieved)
        answer = generate_answer(prompt)

    # Append assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("📚 Retrieved Context"):
            for i, r in enumerate(retrieved):
                st.markdown(f"**{i+1}.** {r['chunk_text'][:300]}...")
