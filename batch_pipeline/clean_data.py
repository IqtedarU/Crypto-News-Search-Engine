import boto3
import gzip
import json
import unicodedata
from bs4 import BeautifulSoup
import os
import html

# Config
bucket = "crypto-search-pipeline-iqtedar"
region = "us-east-1"
prefix_raw = "raw_docs/"
prefix_cleaned = "cleaned_docs/"

s3 = boto3.client("s3", region_name=region)

def clean_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = unicodedata.normalize("NFKD", text)
    return " ".join(text.lower().split())

def clean_documents():
    # Get all raw keys
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix="raw_docs/")

    raw_files = []
    for page in page_iterator:
        raw_files.extend(page.get("Contents", []))

    if not raw_files:
        print("No raw files found.")
        return

    for obj in raw_files:
        raw_key = obj['Key']
        if not raw_key.endswith('.json.gz'):
            continue
        doc_id = os.path.basename(raw_key).replace('.json.gz', '')
        cleaned_key = f"{prefix_cleaned}{doc_id}.json.gz"

        # Skip if already cleaned
        try:
            s3.head_object(Bucket=bucket, Key=cleaned_key)
            print(f"[SKIP] {cleaned_key} already exists.")
            continue
        except s3.exceptions.ClientError:
            pass  # Skip if not found

        # Download and clean
        print(f"[CLEAN] {raw_key} → {cleaned_key}")
        raw_bytes = s3.get_object(Bucket=bucket, Key=raw_key)['Body'].read()
        data = json.loads(gzip.decompress(raw_bytes))

        data["Content"] = clean_text(data.get("Content", ""))
        data["Title"] = clean_text(data.get("Title", ""))

        # Upload cleaned
        cleaned_body = gzip.compress(json.dumps(data).encode("utf-8"))
        s3.put_object(Bucket=bucket, Key=cleaned_key, Body=cleaned_body)

if __name__ == "__main__":
    clean_documents()
