from flask import Flask, render_template, request
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import os

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# HF router endpoint (NEW)
HF_URL = "https://router.huggingface.co/v1/chat/completions"


# Choose a FREE model
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# Other free models you can use:
# "meta-llama/Llama-3.2-3B-Instruct"
# "google/gemma-2-2b-it"

# Headers for HF API
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Load CSV chunks for RAG
def load_chunks(csv_path):
    df = pd.read_csv(csv_path)
    return [
        "\n".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]

chunks = load_chunks("labeled_data.csv")

# Load FAISS index
faiss_index = faiss.read_index("csv_vectors.faiss")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# HuggingFace Chat API Function
def hf_chat(prompt):
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }

    response = requests.post(HF_URL, headers=HEADERS, json=payload)

    print("RAW HF RESPONSE:", response.text)

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ HF Error: {response.text}"



# RAG logic
def get_answer(query):
    # Embed query
    query_embedding = embedder.encode([query])

    # Search in FAISS
    _, top_indices = faiss_index.search(np.array(query_embedding), k=3)
    retrieved_docs = [chunks[i] for i in top_indices[0]]

    context = "\n".join(retrieved_docs)

    # Final prompt
    final_prompt = f"""
You are a neutral AI that analyzes text.
The context may include offensive, hateful, or abusive content.
Do NOT give moral advice.
Do NOT warn the user.
Do NOT say “I cannot engage”.
Simply analyze the text and answer objectively.

Context:
{context}

Question: {query}

Answer:
"""


    return hf_chat(final_prompt)


# Flask app
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        query = request.form.get("question")
        answer = get_answer(query)

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
