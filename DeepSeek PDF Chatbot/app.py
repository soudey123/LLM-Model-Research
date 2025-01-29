import streamlit as st
import concurrent.futures
import json
import pymupdf  # PyMuPDF for rendering PDF images
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA


@st.cache_resource
def get_embedder():
    """Cache the embedding model to avoid reloading on every run."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_pdf(file):
    """Load and process the uploaded PDF file."""
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    return loader.load()


def get_pdf_first_page_image(file):
    """Extract and return the first page of the uploaded PDF as an image."""
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    doc = pymupdf.open("temp.pdf")

    # Ensure 'static/' directory exists
    os.makedirs("static", exist_ok=True)

    pix = doc[0].get_pixmap()
    image_path = "static/first_page.png"
    pix.save(image_path)
    return image_path


def chunk_documents(docs):
    """Efficiently split documents using a recursive text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)


@st.cache_resource
def create_retriever(docs_json):
    """Cache FAISS vector store while ensuring metadata includes 'source'."""
    docs = json.loads(docs_json)  # Deserialize properly into a list of dictionaries

    embedder = get_embedder()
    vector = FAISS.from_texts(
        [doc["page_content"] for doc in docs],
        embedder,
        metadatas=[{"source": doc["metadata"].get("source", "Uploaded PDF")} for doc in docs]
        # Ensure 'source' metadata
    )
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})


@st.cache_resource
def get_llm():
    """Initialize the DeepSeek R1 language model."""
    return Ollama(model="deepseek-r1:1.5b")


def build_prompt():
    """Define the QA prompt template."""
    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
    3. Keep the answer crisp and limited to 3-4 sentences.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    return PromptTemplate.from_template(prompt)


def build_qa_chain(retriever, llm):
    """Build the QA chain using LangChain components."""
    prompt = build_prompt()
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
    )


def process_pdf(file):
    """Process the PDF asynchronously using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_docs = executor.submit(load_pdf, file)
        docs = future_docs.result()

        future_chunks = executor.submit(chunk_documents, docs)
        documents = future_chunks.result()

    # Serialize full document objects instead of just 'page_content'
    docs_json = json.dumps(
        [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    )

    # Create retriever using cached FAISS index
    retriever = create_retriever(docs_json)
    return retriever


def main():
    """Main Streamlit app function with sidebar layout."""
    st.set_page_config(layout="wide")  # Set to wide mode

    st.title("🚀 Fast RAG-based QA with DeepSeek R1")

    with st.sidebar:
        st.subheader("📂 Upload PDF & Model Selection")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file:
            st.subheader("📜 Preview of Uploaded Document")
            try:
                image_path = get_pdf_first_page_image(uploaded_file)
                st.image(image_path, caption="First Page Preview", use_column_width=True)
            except Exception as e:
                st.error("Failed to load preview: " + str(e))

    st.subheader("📝 Ask Questions about the Document")

    if uploaded_file:
        with st.spinner("🔄 Processing PDF..."):
            retriever = process_pdf(uploaded_file)

        llm = get_llm()
        qa_chain = build_qa_chain(retriever, llm)

        user_input = st.text_input("Enter your question:")

        if user_input:
            with st.spinner("🤖 Generating response..."):
                response = qa_chain.invoke({"query": user_input})["result"]
                st.write("### 📜 Answer:")
                st.write(response)
    else:
        st.info("📥 Please upload a PDF file to proceed.")


if __name__ == "__main__":
    main()