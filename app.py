import streamlit as st
import os
import sys


sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure correct path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add parent directory
from  llm_app import retrieved, features, generate_diagnosis,extract_text_from_pdf


st.title("RAG-based LLM for Medical Diagnosis")

#user query 
user_query = st.text_area("Enter your query here:", height=100)

# Medical History of the patient
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

medical_history = None

if uploaded_file is not None:
    #extract text from the uploaded file
    medical_history =  extract_text_from_pdf(pdf_file=uploaded_file)


if st.button("Generate Diagnosis"):

    diagnosis = generate_diagnosis(features, retrieved, user_query, medical_history )


    if user_query:
        with st.spinner("Generating Response"):
        
            st.write("### Response:")
            st.write(diagnosis)
    else:
        st.warning("Please enter a valid question.")