import json
import boto3
import faiss
import numpy as np
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer

s3 = boto3.client('s3')
BUCKET_NAME = "crypto-search-pipeline-iqtedar"
INDEX_KEY = "faiss_index/index.faiss"

model = SentenceTransformer('all-MiniLM-L6-v2')

def lambda_handler(event, context):
    record = event['Records'][0]
    cleaned_key = record['s3']['object']['key']

    if not cleaned_key.startswith("cleaned_docs/"):
        return {"message": "Not a cleaned_docs file, ignoring."}

    # Load cleaned doc
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=cleaned_key)
    doc = json.loads(obj['Body'].read().decode('utf-8'))

    # Create embedding
    text = doc.get('Title', '') + ' ' + doc.get('Content', '')
    embedding = model.encode([text], convert_to_numpy=True).astype('float32')

    # Update FAISS index
    with NamedTemporaryFile() as tmp_index:
        s3.download_file(BUCKET_NAME, INDEX_KEY, tmp_index.name)
        index = faiss.read_index(tmp_index.name)

        index.add(embedding)

        faiss.write_index(index, tmp_index.name)
        s3.upload_file(tmp_index.name, BUCKET_NAME, INDEX_KEY)

    return {"message": f"Added {cleaned_key} to FAISS index"}
