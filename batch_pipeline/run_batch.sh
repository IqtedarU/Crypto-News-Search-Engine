#!/bin/bash

set -e  
set -o pipefail  

echo "Scraping news articles"
python3 batch_pipeline/scrape_news.py

echo "Cleaning raw articles"
python3 batch_pipeline/clean_data.py

echo "Building FAISS index"
python3 batch_pipeline/embed_and_index.py

echo "Batch pipeline completed."
