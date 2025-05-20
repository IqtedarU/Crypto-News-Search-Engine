import gzip
import json
import boto3
import faiss
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

BUCKET_NAME = "crypto-search-pipeline-iqtedar"
S3_PREFIX = "cleaned_docs/"
MODEL_NAME = "all-MiniLM-L6-v2"
USE_AVG_SENTENCES = True
LOCAL_FAISS_PATH = "faiss.index"
LOCAL_DOCMAP_PATH = "doc_id_map.json"
S3_INDEX_KEY = "index/faiss.index"
S3_MAP_KEY = "maps/doc_id_map.json"


s3 = boto3.client("s3")
model = SentenceTransformer(MODEL_NAME)

# List all cleaned docs
response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)
files = []
continuation_token = None

while True:
    if continuation_token:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=S3_PREFIX,
            ContinuationToken=continuation_token
        )
    else:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=S3_PREFIX
        )

    contents = response.get("Contents", [])
    files += [obj["Key"] for obj in contents if obj["Key"].endswith(".json.gz")]

    if response.get("IsTruncated"):
        continuation_token = response.get("NextContinuationToken")
    else:
        break


doc_embeddings = []
doc_map = {}

print(f"Found {len(files)} documents in S3")
for i, key in enumerate(tqdm(files)):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    raw_bytes = obj["Body"].read()
    data = json.loads(gzip.decompress(raw_bytes).decode("utf-8"))

    title = data.get("Title", "")
    content = data.get("Content", "")
    url = data.get("URL", "")

    full_text = f"{title}. {content}".strip()

    if USE_AVG_SENTENCES:
        sentences = sent_tokenize(full_text)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_embeddings = model.encode(sentences)

        weights = np.array(tfidf_matrix.sum(axis=1)).flatten()
        weights /= weights.sum()

        doc_vector = np.average(sentence_embeddings, axis=0, weights=weights)
    else:
        doc_vector = model.encode(full_text)

    doc_embeddings.append(doc_vector)
    doc_map[str(i)] = {
        "s3_key": key,
        "title": title,
        "url": url
    }

dimension = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype("float32"))

faiss.write_index(index, LOCAL_FAISS_PATH)
with open(LOCAL_DOCMAP_PATH, "w") as f:
    json.dump(doc_map, f, indent=2)

s3.upload_file(LOCAL_FAISS_PATH, BUCKET_NAME, S3_INDEX_KEY)
s3.upload_file(LOCAL_DOCMAP_PATH, BUCKET_NAME, S3_MAP_KEY)

print(f"\nUploaded FAISS index to: s3://{BUCKET_NAME}/{S3_INDEX_KEY}")
print(f"Uploaded doc map to: s3://{BUCKET_NAME}/{S3_MAP_KEY}")
