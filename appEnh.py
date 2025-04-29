from turtle import width
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import os
import base64
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
import docx2txt
import openpyxl

# Models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Directory structure
INCIDENT_DIR = os.path.join("project_data", "incidents")
PROJECT_DOC_DIR = os.path.join("project_data", "project_information")
os.makedirs(INCIDENT_DIR, exist_ok=True)
os.makedirs(PROJECT_DOC_DIR, exist_ok=True)

st.set_page_config(page_title="Smart Incident Resolver", layout="wide")

def set_background_image(image_file):
    with open(image_file, "rb") as image:
        encoded_image = image.read()

    st.markdown(
        f"""
    <style>
            .stApp {{
                background-image: url('data:image/jpg;base64,{base64.b64encode(encoded_image).decode()}');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}

            h1, h2, h3, h4, h5, h6, p, label {{
                color: #ffffff;
            }}

            .css-18e3th9 {{
                background-color: rgba(0, 0, 0, 0.6) !important;
                padding: 20px;
                border-radius: 12px;
            }}

            .stTextInput>div>div>input {{
                background-color: #ffffff;
                color: #000000;
            }}

            .stFileUploader>div>div {{
                width: 40% !important;
                padding: 5px;
                background-color: #ffffff;
                color: #000000;
            }}

            .stButton>button {{
                background-color: #ff7e5f;
                color: #fff;
                border-radius: 8px;
            }}

            .stButton>button:hover {{
                background-color: #feb47b;
                color: #000;
            }}

             
            /* Minimize dropdown size */
                .stSelectbox div[data-baseweb="select"] {{
                width: 20% !important; /* Adjust width */
                margin: 0; /* Remove centering */
            }}


            .stFileUploader>div>div {{
                width: 30% !important;  /* ✅ Change from 40% to 30% */
                padding: 5px;
                background-color: #ffffff;
                color: #000000;
                border-radius: 8px;
                align: left;
            }}
    </style>
            """,
        unsafe_allow_html=True
    )

set_background_image('roboimage.jpg')

st.markdown("""
<h1>🤖 AI-Powered Incident Resolution</h1>
<p>Upload incident history or search for smart resolution suggestions 🔍</p>
""", unsafe_allow_html=True)

def load_incident_csv(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(subset=['description', 'resolution'], inplace=True)
    return df

def load_all_incidents():
    all_files = [os.path.join(INCIDENT_DIR, f) for f in os.listdir(INCIDENT_DIR) if f.endswith('.csv')]
    all_dfs = [load_incident_csv(f) for f in all_files]
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=['description', 'resolution'])

def create_index_from_descriptions(descriptions):
    embeddings = embedding_model.encode(descriptions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def search_incidents(query, df, index, embeddings):
    query_vector = embedding_model.encode([query])
    _, indices = index.search(query_vector, k=len(df))
    return df.iloc[indices[0]]

def summarize_texts(texts):
    text = " ".join(texts)
    if len(text.split()) < 10:
        return "Not enough information to summarize."
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".xlsx"):
        wb = openpyxl.load_workbook(file_path)
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                text += " ".join([str(cell) for cell in row if cell is not None]) + "\n"
    return text.strip()

def get_text_chunks(text, chunk_size=100):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def answer_from_docs(query, docs_text):
    if not docs_text.strip():
        return "No project documents found."
    chunks = get_text_chunks(docs_text)
    chunk_embeddings = embedding_model.encode(chunks)
    query_embedding = embedding_model.encode([query])
    index = faiss.IndexFlatL2(len(query_embedding[0]))
    index.add(np.array(chunk_embeddings))
    _, matched_indices = index.search(np.array(query_embedding), k=3)
    top_chunks = [chunks[i] for i in matched_indices[0]]
    combined_context = " ".join(top_chunks)
    try:
        answer = summarizer(f"question: {query} context: {combined_context}", max_length=100, min_length=30, do_sample=False)
        return answer[0]['summary_text']
    except Exception as e:
        return f"Sorry, there was an error generating the response: {e}"

# Main
mode = st.radio("Choose an action:", ["Upload", "Search"])

if mode == "Upload":
    doc_type = st.selectbox("Select document type to upload:", ["Incident data", "Other project documents"])
    if doc_type == "Incident data":
        file = st.file_uploader("📁 Upload your incident_data.csv", type=["csv"], label_visibility="collapsed")
        if file and st.button("Upload"):
            file_path = os.path.join(INCIDENT_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Uploaded successfully.")
    elif doc_type == "Other project documents":
        file = st.file_uploader("Upload Documents", type=["pdf", "docx", "xlsx"])
        if file and st.button("Upload"):
            file_path = os.path.join(PROJECT_DOC_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Uploaded successfully.")

elif mode == "Search":
    search_type = st.selectbox("Search in:", ["Incident data", "Project information"])

    if search_type == "Incident data":
        query = st.text_input("Enter incident description to search:")
        if query and st.button("Search Incidents"):
            df = load_all_incidents()
            if df.empty:
                st.warning("No incident data found.")
            else:  # Exact Keyword Match
                pattern = rf'\b{query.lower()}\b'
                df['description_clean'] = df['description'].str.lower().str.strip()
                results = df[df['description_clean'].str.contains(pattern, regex=True, na=False)]
                if results.empty:
                    st.info("No matching incidents found.")
                else:
                    summary = summarize_texts(results['resolution'].tolist())
                    st.subheader("Matching Results:")
                    st.dataframe(results)
                    st.subheader("Summary:")
                    st.success(summary)

    elif search_type == "Project information":
        sub_option = st.selectbox("Choose", ["Chatbot", "Summary"])
        all_docs_text = ""
        for f in os.listdir(PROJECT_DOC_DIR):
            full_path = os.path.join(PROJECT_DOC_DIR, f)
            all_docs_text += extract_text_from_file(full_path) + "\n"

        if sub_option == "Chatbot":
            prompt = st.text_input("Ask something about the project:")
            if prompt and st.button("Ask"):
                response = answer_from_docs(prompt, all_docs_text)
                st.write(f"🤖 Chatbot says: {response}")

        elif sub_option == "Summary":
            topic = st.text_input("Enter topic to summarize:")
            if topic and st.button("Summarize"):
                if not all_docs_text.strip():
                    st.warning("No uploaded project docs to summarize.")
                else:
                    filtered_text = "\n".join(
                        [line for line in all_docs_text.split("\n") if topic.lower() in line.lower()])
                    summary = summarize_texts([filtered_text]) if filtered_text else "Topic not found in documents."
                    st.success(f"📝 Summary of '{topic}': {summary}")
