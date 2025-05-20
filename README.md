# Crypto News Search Engine

This project is a near real-time search engine for cryptocurrency news articles. It has a hybrid batch and event-driven pipeline built with AWS services, FAISS indexing, and a Flask frontend to support fast document retrieval using vector similarity.

All articles are stored in compressed JSON format in S3 and include fields such as date, time, tag, author, and full article content.

## Project Structure
```
crypto-news-search-engine/
│
├── app/
│   ├── main.py                      # Flask search engine with L2 similarity
│   └── templates/
│       ├── index.html              # Query input form
│       └── search.html             # Displays ranked article results
│
├── batch_pipeline/
│   ├── scrape_news.py              # Scrapes CryptoSlate articles by page
│   ├── clean_data.py               # Normalizes and filters raw scraped JSON
│   ├── embed_and_index.py         # Embeds and adds new articles to FAISS index
│   └── run_batch.sh               # Runs full pipeline: scrape to clean to index
│
├── data/
│   ├── faiss.index                # FAISS vector index for fast search
│   ├── doc_id_map.json           # Maps vector index IDs to article metadata
│   ├── sample_cleaned.json       # Example of cleaned articles
│   └── sample_raw.json           # Example of raw scraped articles
│
├── lambda_functions/
│   ├── cleaned_to_index/
│   │   └── lambda_function.py     # Embeds and adds new article to FAISS index
│   └── raw_to_cleaned/
│       └── lambda_function.py     # Cleans raw article when new file arrives
```

## Setup and Pipeline Overview
### 1. Initial Setup
- Replace all hardcoded S3 bucket names in scripts with your own bucket name.

- Fill out AWS credentials using environment variables or a credential configuration method supported by boto3.

- Run run_batch.sh once to populate your S3 storage with data and initialize the FAISS index.

### 2. Continious Scraping
- Configure the scraper to scan a full page range initially.

- For continuous scraping, reduce to the first few pages and deploy the scraper on an EC2 instance with a fixed time interval (e.g., every 15 minutes).

- This enables periodic capture of recent articles with minimal resource usage.

### 3. AWS Event-Driven Cleaning and Updating Index
- raw_to_cleaned Lambda: Triggered by new raw JSON uploads; cleans and stores the result.

- cleaned_to_index Lambda: Triggered by new cleaned JSON uploads; embeds and adds it to the FAISS index.

### 4. Full Index Rebuild

- For full refresh, schedule embed_and_index.py to run on EC2 at longer intervals.

- This is important especially if you're using indexing methods that do not support dynamic/incremental updates, such as some variants of KNN graph-based or tree-based indexes.

## Flask Search Interface

- Flask provides a lightweight interface to query news articles.

- Embeddings are generated with SentenceTransformer models and compared using L2 distance via FAISS.

- Screenshots of the front page and a example query below:
![FrontPage](https://github.com/user-attachments/assets/61c74120-af0e-4861-bb5a-535229d0ac0b)
![Query](https://github.com/user-attachments/assets/b3fcac9f-1aa1-45ab-9a56-264eedaa5323)


## Data Schema
Example JSON structure for articles:
```
{
  "Date": "Jan. 23, 2023",
  "Time": "12:40 pm",
  "Tag": "Regulation",
  "Author": "Oluwapelumi Adejumo",
  "Free": "False",
  "Title": "australian finance minister says crypto could be regulated as financial product",
  "Content": "...full article content...",
  "URL": "https://cryptoslate.com/australian-finance-minister-says-crypto-could-be-regulated-as-financial-product/"
}
```
