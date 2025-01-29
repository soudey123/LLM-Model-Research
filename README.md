# PDF Question-Answering Application with DeepSeek R1

A Streamlit-based application that enables users to upload PDF documents and ask questions about their content using the DeepSeek R1 language model. The application implements RAG (Retrieval-Augmented Generation) for accurate and context-aware responses.

## 🌟 Features

- PDF document upload and processing
- First page preview of uploaded documents
- Real-time question-answering capabilities
- Fast document processing with concurrent execution
- Efficient document chunking and embedding
- Local vector storage using FAISS
- Integration with DeepSeek R1 language model via Ollama

## 📋 Prerequisites

- Python 3.8+
- Ollama installed with DeepSeek R1 model
- Sufficient RAM for document processing and embeddings

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install streamlit langchain-community pymupdf faiss-cpu sentence-transformers
```

3. Ensure Ollama is installed and the DeepSeek R1 model is pulled:
```bash
ollama pull deepseek-r1:1.5b
```

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Navigate to the provided local URL (typically `http://localhost:8501`)

3. Use the sidebar to upload a PDF document

4. Enter questions about the document content in the text input field

## 💡 How It Works

1. **Document Processing**:
   - PDF documents are loaded using PyPDFLoader
   - Documents are split into chunks using RecursiveCharacterTextSplitter
   - Text chunks are embedded using the MiniLM-L6-v2 model
   - Embeddings are stored in a FAISS vector store

2. **Question Answering**:
   - User questions trigger a similarity search in the vector store
   - Relevant document chunks are retrieved
   - DeepSeek R1 generates answers based on the retrieved context

## 🏗️ Architecture

The application uses several key components:
- `Streamlit`: Web interface
- `LangChain`: Document processing and chain orchestration
- `FAISS`: Vector storage and similarity search
- `HuggingFace Embeddings`: Text embedding generation
- `Ollama`: Local LLM inference
- `PyMuPDF`: PDF rendering and preview generation

## 🔄 Caching

The application implements strategic caching using `@st.cache_resource` for:
- Embedding model
- Vector store retriever
- Language model initialization

## ⚠️ Limitations

- PDF processing time depends on document size
- Memory usage scales with document size
- Requires local installation of Ollama and DeepSeek R1 model
- Limited to text-based question answering

## 🔒 Dependencies

- `streamlit`
- `langchain-community`
- `pymupdf`
- `faiss-cpu`
- `sentence-transformers`
- `ollama`

## 📁 Project Structure

```
.
├── app.py              # Main application file
├── static/            # Directory for temporary image storage
└── temp.pdf           # Temporary storage for uploaded PDF
```
