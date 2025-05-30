import faiss
import json
import gzip
import boto3
from flask import Flask, request, render_template, redirect
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# Set your OpenAI API key here
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FAISS_INDEX_PATH = "c:/users/iqtedar/downloads/faiss.index"
DOC_MAP_PATH = "c:/users/iqtedar/downloads/doc_id_map.json"
MODEL_NAME = "all-MiniLM-L6-v2"
BUCKET_NAME = "crypto-search-pipeline-iqtedar"

# Load FAISS and doc map
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(DOC_MAP_PATH, "r", encoding="utf-8") as f:
    doc_id_map = json.load(f)

# Load sentence-transformers encoder
encoder = SentenceTransformer(MODEL_NAME)

# Connect to S3
s3 = boto3.client("s3")

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('search2.html')

@app.route('/rag', methods=['GET'])
def rag():
    query = request.args.get("query")
    if not query:
        return "Missing query", 400

    query_vec = encoder.encode([query]).astype("float32")
    D, I = faiss_index.search(query_vec, k=5)

    contexts = []
    search_results = []

    for idx in I[0]:
        meta = doc_id_map.get(str(idx))
        if not meta:
            continue
        s3_key = meta["s3_key"]
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        data = json.loads(gzip.decompress(obj["Body"].read()))
        content = data.get("Content", "")
        title = data.get("Title", "")
        url = data.get("URL", "")

        contexts.append(content)
        search_results.append({
            "doc_id": str(idx),
            "title": title,
            "url": url,
        })

    # Build prompt and truncate if needed
    context_text = "\n\n".join(contexts[:3])  # limit to 3 docs for token budget
    prompt = f"""
Answer the user's question based only on the context provided.

Context:
{context_text}

Question:
{query}

Answer:"""

    if len(prompt) > 3500:
        prompt = prompt[:3500]  # crude truncation to avoid token limit errors

    # OpenAI ChatCompletion call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )

    answer = response.choices[0].message.content.strip()

    return render_template("search2.html", query=query, search_results=search_results, generated_answer=answer)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
