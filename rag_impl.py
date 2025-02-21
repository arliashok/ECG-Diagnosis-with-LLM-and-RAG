import numpy as np
import neurokit2 as nk  # For ECG preprocessing
import chromadb  # For embedding storage
from sentence_transformers import SentenceTransformer
import re




# Create Embeddings for Medical Knowledge Base
def create_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings


# Store Embeddings in ChromaDB
def store_embeddings(texts, embeddings):
    #create a random number 
    random_number = np.random.randint(1000)
    client = chromadb.Client()
    collection = client.create_collection(name=f"medical_knowledge_v{random_number}")
    for i, text in enumerate(texts):
        collection.add(ids=[str(i)], documents=[text], embeddings=[embeddings[i].tolist()])
    return collection


#Feature Extraction from ECG
def extract_features(ecg_signal, sampling_rate=100):
    # QRS Duration (for CD)
    r_onsets = ecg_signal["ECG_R_Onsets"].dropna().astype(int)
    r_offsets = ecg_signal["ECG_R_Offsets"].dropna().astype(int)
    qrs_duration = np.nanmean((r_offsets - r_onsets) / sampling_rate * 1000)  # in ms

    # ST Elevation (for MI)
    st_elevations = []
    for s_peak in ecg_signal["ECG_S_Peaks"].dropna().astype(int):
        j_point = s_peak + int(0.08 * sampling_rate)  # J-point 80ms after S-peak
        if j_point < len(ecg_signal):
            st_elevation = ecg_signal["ECG_Clean"].iloc[j_point] - np.mean(ecg_signal["ECG_Clean"])
            st_elevations.append(st_elevation)
    st_elevation = np.nanmax(st_elevations) if st_elevations else 0.0

    # R Wave Amplitude (for HYP)
    r_amplitudes = ecg_signal["ECG_Clean"][ecg_signal["ECG_R_Peaks"].dropna().astype(int)]
    r_wave_v5 = np.nanmax(r_amplitudes)  # Simplified example (lead-specific logic needed)

    # ST/T Changes (for STTC)
    st_depression = np.nanmin(ecg_signal["ECG_ST_Segment"]) if "ECG_ST_Segment" in ecg_signal else 0.0

    return {
        "qrs_duration": qrs_duration,
        "st_elevation": st_elevation,
        "r_wave_v5": r_wave_v5,
        "st_depression": st_depression
    }
   

# Retrieve Similar Cases using RAG
def retrieve_similar_cases(query, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=1)
        # Ensure there's always a status report
    if not results or 'documents' not in results or not results['documents']:
        return {
            "status": "No similar cases found",
            "query": query,
            "results": []
        }

    return results