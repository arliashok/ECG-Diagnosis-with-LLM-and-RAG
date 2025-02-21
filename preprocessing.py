import numpy as np
import neurokit2 as nk  # For ECG preprocessing
import chromadb  # For embedding storage
from sentence_transformers import SentenceTransformer
import requests  # For interacting with Groq LLM
import PyPDF2  # For extracting text from PDFs
import json
import re
import wfdb


def read_raw_ecg():
    record = wfdb.rdrecord('./database/05469_lr')
    
    ecg_signal = record.p_signal
    ecg_channel = ecg_signal[:, 1]
    ecg_cleaned = nk.ecg_clean(ecg_channel, sampling_rate=100)
    
    # 3. Find R-peaks and analyze ECG
    signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=100)
    return signals, info


#Load and Preprocess ECG Data
def preprocess_ecg(raw_signal, sampling_rate=100):
    ecg_cleaned = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate)
    return ecg_cleaned


#Extract and  Preprocess Text from PDFs
def extract_text_from_pdf(pdf_paths=None,pdf_file=None, chunk_size=500):
    text = ""

    if pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += re.sub(r'\s+', ' ', page_text.strip()) + " "

    else:

        for pdf_path in pdf_paths:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += re.sub(r'\s+', ' ', page_text.strip()) + " "
        
    # Chunking text into manageable parts
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

