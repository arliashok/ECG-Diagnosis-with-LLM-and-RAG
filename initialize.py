from llm_app import create_embeddings,extract_features, retrieve_similar_cases, store_embeddings,extract_text_from_pdf, preprocess_ecg, read_raw_ecg
import streamlit as st

@st.cache_resource 
def initialize_pipeline():
    
    # Extract and Preprocess Text from PDFs
    pdf_paths = ["./database/paper03.pdf", "./database/paper04.pdf"]
    text_chunks = extract_text_from_pdf(pdf_paths, chunk_size=500)
    
    # Create Embeddings for Medical Knowledge Base
    embeddings = create_embeddings(text_chunks)
    
    # Store Embeddings in ChromaDB
    collection = store_embeddings(text_chunks, embeddings)

    #read raw ecg from the source
    ecg_signal, info = read_raw_ecg()    

    features = extract_features(ecg_signal)

    retrieved = retrieve_similar_cases(str(features), collection)
    
    return retrieved, features


# Initialize the pipeline
retrieved, features = initialize_pipeline()