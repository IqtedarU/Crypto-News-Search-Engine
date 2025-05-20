import json
import boto3
import html
import unicodedata

s3 = boto3.client('s3')
BUCKET_NAME = "crypto-search-pipeline-iqtedar"

def clean_text(text):
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    return ' '.join(text.split())

def lambda_handler(event, context):
    record = event['Records'][0]
    raw_key = record['s3']['object']['key']

    if not raw_key.startswith("raw_docs/"):
        return {"message": "Not a raw_docs file, ignoring."}

    # Load raw content
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=raw_key)
    raw_data = json.loads(obj['Body'].read().decode('utf-8'))

    # Clean fields
    cleaned_data = {
        "Title": clean_text(raw_data.get("title", "")),
        "Content": clean_text(raw_data.get("content", "")),
        "Author": clean_text(raw_data.get("author", "")),
        "Tag": clean_text(raw_data.get("tag", "")),
        "Date": clean_text(raw_data.get("date", "")),
        "Time": clean_text(raw_data.get("time", "")),
        "URL": raw_data.get("url", ""),
        "Free": str(raw_data.get("free", "False"))
    }

    # Save to cleaned_docs with same filename
    cleaned_key = raw_key.replace("raw_docs/", "cleaned_docs/")
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=cleaned_key,
        Body=json.dumps(cleaned_data),
        ContentType='application/json'
    )

    return {"message": f"Cleaned and stored: {cleaned_key}"}
