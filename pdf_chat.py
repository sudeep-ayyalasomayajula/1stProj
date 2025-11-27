import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📚 Chat with your PDF (RAG System)")

# 2. API SETUP (Safe Handling)
# We check if the key is in Streamlit secrets; otherwise, we ask for it.
api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.text_input("Enter Google API Key:", type="password")

if api_key:
    # Configure the standard Google library
    genai.configure(api_key=api_key)

    # 3. PROCESSING THE PDF (The "Ingestion" Phase)
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        # Step A: Read the PDF text
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        st.info(f"PDF Loaded successfully! Total characters: {len(text)}")

        # Step B: Split text into "Chunks" (Index Cards)
        # We don't feed 500 pages to the AI. We split it into 1000-character chunks.
        # 'overlap' ensures we don't cut a sentence in half weirdly.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # Step C: Create Embeddings & Vector Store (The "Filing Cabinet")
        # This converts text chunks into numbers (vectors) using Google's embedding model.
        # 'FAISS' stores these numbers for fast searching.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # Standard stable model
            google_api_key=api_key
        )
        
        # This line actually builds the searchable database in your computer's RAM
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.success("Analysis Complete! Ask a question below.")

        # 4. HANDLING QUESTIONS (The "Retrieval" Phase)
        user_question = st.text_input("Ask a question about this PDF:")

        if user_question:
            # Step D: Setup the "Librarian" (The Retriever)
            # We tell the vector store to act as a retriever looking for the top 5 relevant chunks
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})

            # Step E: Setup the "Chef" (Gemini 2.5)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3, # Low temperature = more factual, less creative
                google_api_key=api_key
            )

            # Step F: The "Prompt" that binds it all together
            # {context} is where the retrieved chunks will be injected automatically
            prompt_template = ChatPromptTemplate.from_template("""
            Answer the question based ONLY on the following context:
            {context}

            Question: {input}
            """)

            # Step G: Build the Chain (The Pipeline)
            # 1. 'stuff_documents_chain' puts the retrieved chunks into the prompt
            # 2. 'retrieval_chain' manages the whole flow: Question -> Search -> Answer
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Step H: Run it!
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({"input": user_question})
                st.write(response["answer"])

            # OPTIONAL: Show sources (What chunks did it use?)
            with st.expander("See retrieved context (Debug)"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1}:** {doc.page_content}")