import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="Multilingual RAG Web App", page_icon="🤖", layout="wide")
st.title("Multilingual Chatbot")

# Loading environmental variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is not set in the environment variables")
    st.stop()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Loading the pdf document
file_path = "../document/HSC26-Bangla1st-Paper.pdf"

if st.button("Initialize RAG System"):
    with st.spinner("Loading and processing documents..."):
        try: 
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except FileNotFoundError:
            st.error("PDF file not found")
            st.stop()
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            st.stop()

        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  
            chunk_overlap=150,
            separators=["\n\n", "\n", "।", "।", " ", ""]  
        )
        chunks = text_splitter.split_documents(documents)

        # Using multilingual embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}  
        )

        # Creating vectorstore
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success("RAG system initialized successfully!")
        st.info(f"Processed {len(chunks)} text chunks")

if st.session_state.vectorstore is not None:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can answer questions in Bengali and English. 
        Use the provided context to answer the question accurately. 
        If the question is in Bengali, try to answer in Bengali.
        IMPORTANT: If you cannot find the answer in the provided context, say 'আমি নিশ্চিত নই' (I'm not sure) and don't make up an answer."""),
        ("user", "Question: {question}\nContext: {context}")
    ])
    
    chain = prompt | llm

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}  # Get top 5 most relevant chunks
                    )
                    docs = retriever.invoke(question)
                    
                    # Debug: Show similarity scores
                    st.write("**Debug: Retrieved Documents**")
                    for i, doc in enumerate(docs):
                        st.write(f"Document {i+1} preview: {doc.page_content[:100]}...")
                    
                    # Combine all retrieved content
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    response = chain.invoke({
                        "question": question,
                        "context": context
                    })
                    
                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.subheader("Answer")
        if 'last_response' in st.session_state:
            st.write(st.session_state.last_response)
            
            with st.expander("Show Retrieved Context"):
                for i, doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f"**Relevant Document {i}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")