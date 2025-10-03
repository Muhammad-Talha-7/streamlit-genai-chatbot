from dotenv import load_dotenv
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load env variables
load_dotenv()

# Streamlit page setup
st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="centered",
)

st.title("💬 Generative AI ChatBot")

# Sidebar features
st.sidebar.header("⚙️ Settings")

# Model selector
model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
)

# Creativity slider
temperature = st.sidebar.slider(
    "Creativity (Temperature):", 0.0, 1.0, 0.0, 0.1
)

# Reset chat button
if st.sidebar.button("🗑️ Reset Conversation"):
    st.session_state.chat_history = []

# Download chat history
if st.sidebar.button("💾 Download Chat History"):
    if "chat_history" in st.session_state and st.session_state.chat_history:
        chat_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}"
                               for msg in st.session_state.chat_history])
        st.download_button("📥 Save Chat as TXT", chat_text, file_name="chat_history.txt")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# LLM init
llm = ChatGroq(
    model=model_choice,
    temperature=temperature,
    api_key=os.getenv("GROQ_API_KEY")
)

# File upload
uploaded_file = st.file_uploader("📂 Upload a TXT or PDF file", type=["txt", "pdf"])
retriever = None

if uploaded_file:
    file_content = ""
    if uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            file_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            st.error("Error reading PDF: " + str(e))

    if file_content:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(file_content)

        # Convert into LangChain docs
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Embeddings + FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Input box
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # System prompt
    system_prompt = "You are a helpful assistant."

    # If retriever available, fetch context from file
    if retriever:
        retrieved_docs = retriever.get_relevant_documents(user_prompt)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt += f"\nUse this context from the uploaded file to help answer:\n{context}"

    # Model call
    response = llm.invoke(
        input=[{"role": "system", "content": system_prompt},
               *st.session_state.chat_history]
    )

    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
