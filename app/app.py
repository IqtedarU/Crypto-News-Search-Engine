import time
from flask import Flask, request, jsonify, render_template, send_file, redirect
import faiss
import json
import boto3
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer

BUCKET_NAME = "crypto-search-pipeline-iqtedar"
INDEX_KEY = "faiss_index/index.faiss"
DOC_MAP_KEY = "faiss_index/doc_id_map.json"
MODEL_NAME = "all-MiniLM-L6-v2"

s3 = boto3.client("s3")

model = SentenceTransformer(MODEL_NAME)

with NamedTemporaryFile() as tmp_index:
    s3.download_file(BUCKET_NAME, INDEX_KEY, tmp_index.name)
    faiss_index = faiss.read_index(tmp_index.name)

# Download and load doc ID map
with NamedTemporaryFile(mode='r+', encoding='utf-8') as tmp_map:
    s3.download_file(BUCKET_NAME, DOC_MAP_KEY, tmp_map.name)
    doc_id_map = json.load(tmp_map)

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/view_document/<string:doc_id>')
def view_document(doc_id):
    meta = doc_id_map.get(doc_id)
    if not meta:
        return "Document not found", 404
    return redirect(meta["url"])

@app.route('/search', methods=['GET'])
def search():
    # Extract the search query from the request parameters
    query = request.args.get('query')

    query_vec = model.encode([query]).astype("float32")
    D, I = faiss_index.search(query_vec, k=15)

    search_results = []
    for idx, score in zip(I[0], D[0]):
        print("Indices:", I[0])
        print("Distances:", D[0])

        meta = doc_id_map[str(idx)]
        search_results.append({
            "doc_id": str(idx),
            "title": meta["title"],
            "url": meta["url"],
            "similarity": float(score)
        })

    # Slice search_results to only include the first 15 results
    search_results = search_results[:15]

    # Render the search template with the search results
    return render_template('search.html', query=query, search_results=search_results)


if __name__ == '__main__':
    app.run(debug=True)
    app.run('0.0.0.0', '5000')
