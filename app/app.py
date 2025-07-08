# app.py
import streamlit as st
from rag_pipeline import (
    load_embed_model,
    load_vector_store,
    load_client,
    retrieve_chunks,
    build_prompt,
    generate_answer,
)

st.set_page_config(page_title="CrediTrust Complaint Chat")
st.title("📊 CrediTrust Complaint RAG Chatbot")

# Load resources (cached)
embed_model = load_embed_model()
print("Embed model loaded successfully.")
index, metadata = load_vector_store()
print("Vector store loaded successfully.")
client = load_client()
print("HuggingFace Inference Client initialized successfully.")

# Initialize session state for chat history 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input
if user_query := st.chat_input("Ask a question about customer complaints..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        retrieved = retrieve_chunks(user_query, embed_model, index, metadata)
        prompt = build_prompt(user_query, retrieved)
        answer = generate_answer(prompt, client)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("📚 Retrieved Context"):
            for i, r in enumerate(retrieved):
                st.markdown(f"**{i+1}.** {r['chunk_text'][:300]}...")

if st.button("🔁 Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
