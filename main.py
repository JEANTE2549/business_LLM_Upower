import streamlit as st
from huggingface_hub import InferenceClient

# ----- CONFIG -----
HF_TOKEN = "your_hf_token_here"  # Paste your Hugging Face API token here
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# ----- SESSION STATE -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----- STREAMLIT UI -----
st.set_page_config(page_title="Business Chatbot", page_icon="💼")
st.title("💼 Business Q&A Chatbot")
st.caption("Powered by Zephyr LLM from Hugging Face")

# ----- INPUT -----
user_input = st.text_input("Ask a question about your business:", key="input")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        # Construct prompt history
        history = ""
        for user, bot in st.session_state.chat_history:
            history += f"<|user|>\n{user}\n<|assistant|>\n{bot}\n"
        history += f"<|user|>\n{user_input}\n<|assistant|>\n"

        # Send to model
        response = client.text_generation(
            prompt=history,
            max_new_tokens=256,
            temperature=0.7,
            stop_sequences=["<|user|>"]
        )

        # Save history
        st.session_state.chat_history.append((user_input, response.strip()))

# ----- DISPLAY CHAT -----
st.markdown("### 📜 Conversation History")
for user, bot in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")
    st.markdown("---")
