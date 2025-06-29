########## streamlit run app.py ###################


import os
import tempfile
from typing import List, Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-qa-index"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def extract_text_from_pdf(pdf_file) -> tuple[str, List[Document]]:
    text = ""
    documents = []
    pdf_reader = PdfReader(pdf_file)
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            text += page_text
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": pdf_file.name,
                    "page": page_num,
                    "text": page_text
                }
            )
            documents.append(doc)
    
    return text, documents

def create_vector_store(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(documents)
    
    try:
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes]
        
        if INDEX_NAME not in index_names:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        docsearch = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        
        return docsearch
        
    except Exception as e:
        st.error(f"Error creating/accessing Pinecone index: {str(e)}")
        st.stop()

def get_qa_chain():
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        st.stop()

def summarize_text(text: str) -> str:
    prompt = f"""
    Please provide a detailed summary of the following text. 
    Focus on key points, main ideas, and important details.
    
    Text: {text[:15000]}
    
    Summary:
    """
    
    response = llm.invoke(prompt)
    return response.content

def main():
    st.set_page_config(page_title="PDF Q&A with Summarization", page_icon="📄")
    st.title("PDF Q&A with Summarization")
    st.write("Upload PDFs, get summaries, and ask questions about the content.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_documents = []
        
        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    text, documents = extract_text_from_pdf(tmp_file)
                    all_documents.extend(documents)
                
                with st.expander(f"Summary for {uploaded_file.name}"):
                    summary = summarize_text(text)
                    st.write(summary)
        
        if all_documents:
            with st.spinner("Indexing documents..."):
                create_vector_store(all_documents)
            st.success("Documents processed and indexed successfully!")
            
            st.subheader("Ask a question about the documents")
            question = st.text_input("Your question:")
            
            if question:
                with st.spinner("Searching for answers..."):
                    qa_chain = get_qa_chain()
                    result = qa_chain({"query": question})
                    
                    st.subheader("Answer:")
                    st.write(result["result"])
                    
                    st.subheader("Sources:")
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.write(f"{i}. {doc.metadata['source']} - Page {doc.metadata['page']}")
                        with st.expander(f"View context {i}"):
                            st.text(doc.page_content)
    else:
        st.info("Please upload one or more PDF files to get started.")

if __name__ == "__main__":
    main()