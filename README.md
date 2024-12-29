📚 PDF-Based RAG System with Llama3

This project is a Retrieval-Augmented Generation (RAG) system built using Streamlit, LangChain, and Llama3 via ChatGroq API. It allows users to upload PDF files, process them into vector embeddings using FAISS, and interact with the content using natural language queries.

🚀 Features

Upload PDF Files: Users can upload PDF documents.

Vector Embeddings: Text is chunked and embedded using sentence-transformers/all-mpnet-base-v2.

FAISS Indexing: Efficient search through the embedded vectors.

Interactive QA: Ask questions based on the PDF content.

Chat History: View and maintain chat context across the session.

Reset Vectorstore: Clear existing vectorstore and start fresh.

🛠️ Technologies Used

Python

Streamlit

LangChain

FAISS

HuggingFace Embeddings

ChatGroq API (Llama3 model)

PyPDFLoader

Pickle

dotenv

📦 Installation

Clone the Repository:

git clone https://github.com/your-repo/pdf-rag-system.git
cd pdf-rag-system

Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Set Environment Variables:
Create a .env file in the root folder and add:

API_KEY=your_chatgroq_api_key

▶️ Running the App

Start the Streamlit app:

streamlit run app.py

Access the app at: http://localhost:8501

📄 Usage

Upload a PDF File: Use the sidebar to upload a PDF.

Process PDF: Click the Process PDF button to create embeddings and vectorstore.

Ask Questions: Enter your questions in the chat interface.

Clear Chat: Reset the conversation history if needed.

Delete Vectorstore: Clear vectorstore data via the sidebar.

🛡️ Environment Variables

Ensure the following environment variables are set:

API_KEY=your_chatgroq_api_key

📝 Logs

Application logs are saved in:

../app_checkpoints.log

🐞 Troubleshooting

API Key Missing: Ensure the .env file is correctly configured.

Vectorstore Issues: Use the Delete Vectorstore button to reset.

Device Issues: If GPU is unavailable, the app will fallback to CPU.

🤝 Contributing

Feel free to open issues or create pull requests for improvements.

📧 Contact

For any questions or support, please reach out to:

Email: rahulyadav5122000@gmail.com


📜 License

This project is licensed under the MIT License.

Enjoy exploring your PDFs with AI! 🚀

