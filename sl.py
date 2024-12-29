import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import pickle
import os
import logging
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Initialize QA Chain and previous upload state in session_state if not already done
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "uploaded_pdf_name" not in st.session_state:
    st.session_state.uploaded_pdf_name = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Store conversation history

VECTORSTORE_PATH = "vector_index.pkl"


# Helper function to check if vectorstore is empty
def is_vectorstore_empty():
    return os.path.getsize(VECTORSTORE_PATH) == 0 if os.path.exists(VECTORSTORE_PATH) else True


# Reset vectorstore
def reset_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        os.remove(VECTORSTORE_PATH)
        logging.info("Existing vectorstore deleted before creating a new one.")


if "is_vectorstore_valid" not in st.session_state:
    st.session_state.is_vectorstore_valid = os.path.exists(VECTORSTORE_PATH) and not is_vectorstore_empty()
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Logging setup
logging.basicConfig(
    filename='../app_checkpoints.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="PDF-Based RAG System", layout="wide")

st.title("PDF-Based RAG System with Llama3")
logging.info("Checkpoint 1: Streamlit App launched successfully.")

# Sidebar for uploading and processing the PDF
st.sidebar.header("Vectorstore Management")

if st.session_state.is_vectorstore_valid:
    if st.sidebar.button("Delete Vectorstore", help="This will delete the current vectorstore and reset the app state"):
        reset_vectorstore()
        st.session_state.qa_chain = None
        st.session_state.is_vectorstore_valid = False
        st.session_state.uploaded_pdf_name = None
        st.session_state.pdf_processed = False
        st.session_state.conversation_history = []  # Clear conversation history
        st.sidebar.success("Vectorstore has been deleted successfully! Please upload and process a PDF.")
        logging.info("Checkpoint: Vectorstore manually deleted by user.")
else:
    st.sidebar.info("No valid vectorstore found. Please upload and process a PDF first.")
    logging.info("Checkpoint: No vectorstore found during delete attempt.")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else {}


def load_llm():
    try:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API key is missing.")
        llm = ChatGroq(
            temperature=0,
            api_key=api_key,
            model="llama-3.1-70b-versatile"
        )
        logging.info("Checkpoint 10: Llama3 model (via ChatGroq) initialized successfully.")
        return llm
    except Exception as e:
        st.error(f"Error loading the Llama3 model: {e}")
        logging.error(f"Error loading the Llama3 model: {e}")
        return None


# ✅ Check if a new PDF is uploaded
if uploaded_file:
    st.sidebar.write("File uploaded successfully!")
    logging.info("Checkpoint 2: PDF file uploaded successfully.")

    # Check if the uploaded file is new
    if st.session_state.uploaded_pdf_name != uploaded_file.name:
        st.session_state.uploaded_pdf_name = uploaded_file.name
        reset_vectorstore()
        st.session_state.is_vectorstore_valid = False
        st.session_state.qa_chain = None
        logging.info("Checkpoint 2.1: Detected new PDF upload. Vectorstore reset.")

    if st.sidebar.button("Process PDF", help="Process the uploaded PDF and create a vectorstore"):
        with st.spinner("Processing PDF..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()

            try:
                # Stage 1: Save PDF
                progress_text.write("🔄 Saving PDF locally...")
                with open("../uploaded.pdf", "wb") as f:
                    f.write(uploaded_file.read())
                logging.info("Checkpoint 3: PDF file saved locally.")
                progress_bar.progress(15)

                # Stage 2: Load PDF content
                progress_text.write("📚 Loading PDF content...")
                loader = PyPDFLoader("../uploaded.pdf")
                documents = loader.load()
                logging.info("Checkpoint 4: PDF content loaded successfully.")
                progress_bar.progress(30)

                # Stage 3: Split PDF into chunks
                progress_text.write("✂️ Splitting PDF into chunks...")
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
                docs = text_splitter.split_documents(documents)
                logging.info("Checkpoint 5: PDF content split into chunks.")
                progress_bar.progress(45)

                # Stage 4: Clean document chunks
                progress_text.write("🧹 Cleaning document chunks...")
                cleaned_docs = [d.page_content for d in docs if d.page_content and isinstance(d.page_content, str)]
                logging.info(f"Checkpoint 6: Cleaned documents. Number of valid chunks: {len(cleaned_docs)}")
                progress_bar.progress(60)

                # Stage 5: Generate embeddings and FAISS index
                progress_text.write("🔍 Generating embeddings and creating FAISS index...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs=model_kwargs
                )
                vectorstore = FAISS.from_texts(cleaned_docs, embeddings)
                logging.info("Checkpoint 7: Embeddings generated and FAISS vectorstore created.")
                progress_bar.progress(80)

                # Stage 6: Save vectorstore
                with open(VECTORSTORE_PATH, "wb") as f:
                    pickle.dump(vectorstore, f)
                logging.info("Checkpoint 8: FAISS vectorstore saved successfully.")
                progress_text.write("💾 VectorStore Saved")
                progress_bar.progress(100)

                st.success("PDF processed and vectorstore created successfully!")
                st.session_state.is_vectorstore_valid = True
                st.session_state.pdf_processed = True

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                logging.error(f"Error during PDF processing: {e}")
                st.session_state.is_vectorstore_valid = False

# ✅ Initialize QA Chain if vectorstore exists
if st.session_state.is_vectorstore_valid:
    try:
        with open(VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
        logging.info("Checkpoint 9: Vectorstore loaded successfully.")

        llm = load_llm()

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )
        logging.info("Checkpoint 11: QA Chain created successfully.")

    except Exception as e:
        st.error(f"Error loading the vectorstore or QA chain: {e}")
        logging.error(f"Error in QA Interaction: {e}")
        st.session_state.qa_chain = None
        st.session_state.is_vectorstore_valid = False

# ✅ Question-Answer Interaction
st.header("Chat with the LLM 🤖", anchor="chat-section")

query = st.chat_input("Type your question here...")

if query:
    if not st.session_state.is_vectorstore_valid:
        st.warning("Vectorstore is not valid. Please upload and process a PDF first.")
    elif st.session_state.qa_chain is None:
        st.warning("QA Chain is not initialized. Please process a PDF first.")
    else:
        try:
            response = st.session_state.qa_chain.run(query)

            # Append the user query and the answer to the conversation history
            st.session_state.conversation_history.append({"role": "user", "content": query})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

            # Display all previous questions and answers
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(f"**You:** {message['content']}")
                elif message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.write(f"**Answer:** {message['content']}")
        except Exception as e:
            st.error(f"Error generating the response: {e}")

# Button to clear chat history
if st.button("Clear Chat", help="Clear the conversation history"):
    st.session_state.conversation_history = []
