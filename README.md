# Replication of Zero-Shot ECG Diagnosis with Large Language Models and Retrieval-Augmented Generation: ECG Diagnosis Using Feature Extraction, Knowledge Retrieval, and LLM Inference

This project aims to replicate and extend the methodology presented in [Paper] (https://proceedings.mlr.press/v225/yu23b/yu23b.pdf).
The implementation integrates ECG feature extraction, retrieval-augmented generation (RAG), and large language model (LLM) inference to diagnose ECG abnormalities.

We closely follow the original paper's approach while making necessary adaptations for implementation. Architecture diagrams, methodology breakdowns, and code references are included to facilitate reproducibility.

## 1. Overview
This pipeline integrates ECG feature extraction, knowledge retrieval, and LLM-based diagnosis. The key steps include:

Construction of a domain knowledge database using text embeddings.
ECG feature extraction using Neurokit2 and embedding conversion.
Retrieval-Augmented Generation (RAG) for feature-based query selection.
LLM inference for AI-driven ECG diagnosis.
The architecture of this pipeline is illustrated below:

![image alt](https://github.com/godbright/ECG-Diagnosis-with-LLM-and-RAG/blob/8fef990d1e4a4c7def37e71d0bd4c3c97ccd8cb7/Screenshot%202025-02-21%20at%2021.12.36.png)


## 2. Pipeline Breakdown

### Step 1: Construction of Domain Knowledge Database
 Goal: Create a knowledge base of ECG-related research papers and textbooks for retrieval-based learning.

Process:

Collect ECG-related literature from papers and books.
Extract text embeddings using an embedding model.
Store embeddings in ChromaDB for retrieval.

Key Components:

Database (Papers/Books) → Contains medical literature on ECG.
Text Embedding Extractor → Converts text into embeddings.
ChromaDB → Stores embeddings for retrieval.

### Step 2: Feature Extraction and Prompt Creation
  Goal: Extract ECG features and generate structured prompts for LLM inference.

  Process:

Input raw ECG data from patient recordings.
Use Neurokit2 to extract key waveform features (P-wave, QRS complex, etc.).
Convert ECG features into embeddings for similarity-based retrieval.
Select relevant medical knowledge based on ECG feature embeddings.
Generate a combined prompt (ECG features + retrieved knowledge).

Key Components:

Neurokit2 Feature Extractor → Extracts ECG waveform components.
Text Embedding Extractor → Converts features into embeddings.
Feature Selection → Retrieves similar cases from ChromaDB.
User/System Prompt Generator → Merges user input and extracted knowledge.


### Step 3: LLM Inference for Diagnosis
  Goal: Use an LLM to generate ECG interpretations based on structured prompts.

  Process:

Feed the retrieved knowledge and ECG features into the LLM.
The LLM generates diagnostic interpretations.
Output is evaluated against expert-labeled ECG reports.
  
  Key Components:

LLM Model (e.g., GPT-4, Med-PaLM) → Generates ECG diagnoses.
Prompting System → Structures queries based on RAG.
Evaluation Metrics → Accuracy, Sensitivity, Specificity, F1-score.

## 3. Code Implementation

### Project Structure


📂 LLM_APP  
 ├── 📂 __pycache__/              # Compiled Python files  
 ├── 📂 database/                 # ECG data and medical literature  
 │   ├── 05469_lr.dat             # ECG signal data  
 │   ├── 05469_lr.hea             # ECG header file  
 │   ├── paper03.pdf              # Research paper  
 │   ├── paper04.pdf              # Research paper  
 ├── __init__.py                  # Module initializer  
 ├── .env                         # Environment variables  
 ├── .gitignore                   # Git ignore file  
 ├── app.py                        # Main application entry point  
 ├── config.py                     # Configuration settings  
 ├── initialize.py                 # Initializes project components  
 ├── llm_service.py                # Handles interaction with LLM  
 ├── preprocessing.py               # ECG feature extraction module  
 ├── rag_impl.py                    # Retrieval-Augmented Generation (RAG) implementation  
 ├── README.md                      # Documentation  
 ├── requirements.txt               # Dependencies  

``

## 4. How to Run
### Step 1: Install Dependencies

```bash  
pip install -r requirements.txt
```


### Step 2: Launch the LLM interface using streamlit 

```python
streamlit run app.py 
```



## 4. References
Zero-Shot ECG Diagnosis with Large Language Models and
Retrieval-Augmented Generation - [Paper] (https://proceedings.mlr.press/v225/yu23b/yu23b.pdf).
Related research on ECG feature extraction, RAG, and medical AI.
