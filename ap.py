import gradio as gr  
import faiss
import numpy as np  
import pandas as pd 
from sentence_transformers import SentenceTransformer 
from dotenv import load_dotenv
from together import Together 
import os

# Load API key from .env  
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")  

# Load CSV text chunks
def load_chunks(csv_path):
    df = pd.read_csv(csv_path)
    return [
        "\n".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]  

chunks = load_chunks("labeled_data.csv") 

# Load FAISS index
index = faiss.read_index("csv_vectors.faiss")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Together client
client = Together(api_key=TOGETHER_API_KEY)

# RAG logic
def answer_question(query):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    retrieved = [chunks[i] for i in I[0]]

    context = "\n".join(retrieved)
    final_prompt = f"""Use the context below to answer the question:

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
gr.Interface( 
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question", placeholder="e.g. What is the tweet sentiment?", lines=2),
    outputs=gr.Textbox(label="Answer"),
    title="📊 CSV RAG Chatbot",
    description="Ask questions about your CSV data using Together AI + FAISS + Sentence Transformers.",
    theme="soft",allow_flagging="never"
).launch() 
