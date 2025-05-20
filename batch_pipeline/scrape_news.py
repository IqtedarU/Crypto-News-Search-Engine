import logging
import boto3 as boto3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import hashlib
import gzip


# Hash URL which is unique
def get_doc_id(url):
    return str(int(hashlib.sha1(url.encode()).hexdigest(), 16) % (10 ** 9))

def scrape_articles():


    region_name = 'us-east-1'
    s3 = boto3.client('s3', region_name='us-east-1')

    crypto_slate_url = 'https://cryptoslate.com/news/'

    # Webscrape for cryptoslate
    page_number = 1
    while True:
       try:
        # Open page of news article
        page_url = f'{crypto_slate_url}/page/{page_number}/' if page_number > 1 else crypto_slate_url
        response = requests.get(page_url)
        if response.status_code != 200:
           print('error')
           page_number = page_number+1
           continue

        # Get area where News data are listed
        soup = BeautifulSoup(response.content, 'html.parser')
        # parse to get to featured news
        list_cards = soup.find_all('div', class_='list-card')
       # Go through Individual News Articles on Page
        for list_post in list_cards:
          try:
           title_element = list_post.find('h2')
           title = title_element.text.strip()
           # Extract the URL from the 'href' attribute of the 'a' tag
           article_url = list_post.find('a').get('href')
           # Find the tag and date within the 'post-meta' div
           post_meta = list_post.find('span', class_='post-meta')
           if post_meta is None:
               print(f"Missing post-meta for article: {title}")
               continue
           span_elements = post_meta.find_all('span')
           tag = None
           # Loop through the span elements and extract the tag
           for i, span in enumerate(span_elements):
               tag_text = span.text.strip()
               if tag_text.startswith('Contributor'):
                   if i + 1 < len(span_elements):
                       tag = span_elements[i + 1].text.strip()
                   break
               else:
                   tag = span_elements[0].text.strip()
           if(tag == 'Ad'): # not keeping any Ads
               continue
           else:
               # Go to article page
               response2 = requests.get(article_url)
               if response2.status_code != 200:
                   print('error')
                   continue

               soup = BeautifulSoup(response2.content, 'html.parser')
               article_top_element = soup.find(id='top')
               article_main = article_top_element.find(id='main')
               article_post_container = article_main.find(class_ = 'post-container')
               author_box = article_post_container.find(class_ = 'post-meta-single')
               author_info = author_box.find(class_='author-info')
               author = author_info.find("a").text
               date_time = author_info.find(class_ = 'post-date').text.split()  # Join components 3-5 as the date
               date, time = " ".join(date_time[0:3]), " ".join(date_time[4:6])
               article_post = article_post_container.find(class_ = 'post')
               article_post_content = article_post.find(class_ = 'post-box clearfix')
               if (article_post_content != None):
                    post_content = article_post_content.find('article', class_='full-article')
                    text_content = ''
                    for child in post_content.children:
                       if child.name == 'p':
                           # Extract text from <p> elements
                           text_content += child.get_text(separator=' ',
                                                          strip=True) + ' '  # Add space after each <p> element
                       elif child.name == 'blockquote':
                           # Extract text from <blockquote> elements
                           text_content += child.get_text(separator=' ',
                                                          strip=True) + ' '  # Add space after each <blockquote> element
                       elif child.name == 'a':
                           # Extract text from linked elements
                           linked_text = child.get_text(separator=' ', strip=True)
                           text_content += linked_text + ' '  # Add space after each linked element
                    item = {
                        'Date': date,
                        'Time': time,
                        'Tag': tag,
                        'Author': author,
                        'Free': 'False',
                        'Title': title,
                        'Content': text_content,
                        'URL': article_url

                    }

                    doc_id = get_doc_id(article_url)
                    s3_key = f"raw_docs/{doc_id}.json.gz"

                    # Step 2: Check for existence
                    try:
                        s3.head_object(Bucket="crypto-search-pipeline-iqtedar", Key=s3_key)
                        print(f"{s3_key} already exists. Skipping.")
                        continue
                    except s3.exceptions.ClientError:
                        pass  # Not found — good to save

                    compressed = gzip.compress(json.dumps(item).encode("utf-8"))

                    s3.put_object(
                        Bucket="crypto-search-pipeline-iqtedar",
                        Key=s3_key,
                        Body=compressed
                    )
                    print(f"Uploaded new article: {title}")

               elif (article_post_content == None):
                article_post_content = article_post.find(class_='post-box clearfix cs-box') #Premium Content
                post_content = article_post_content.find('article')
                text_content = ''
                for child in post_content.children:
                   if child.name == 'p':
                       # Extract text from <p> elements
                       text_content += child.get_text(separator=' ',
                                                      strip=True) + ' '  # Add space after each <p> element

                #If item with the same URL already exists, skip inserting it
                if 'Item' in response:
                    #print(f"Article with URL '{url}' already exists. Skipping insertion.")
                    continue
                item = {
                   'Date': date,
                   'Time': time,
                   'Tag': tag,
                    'Author': author,
                   'Free': 'True',
                   'Title': title,
                   'Content': text_content,
                   'URL': article_url

                }

                doc_id = get_doc_id(article_url)
                s3_key = f"raw_docs/{doc_id}.json.gz"

                # Step 2: Check for existence
                try:
                    s3.head_object(Bucket="crypto-search-pipeline-iqtedar", Key=s3_key)
                    print(f"{s3_key} already exists. Skipping.")
                    continue
                except s3.exceptions.ClientError:
                    pass  # Not found — good to save

                compressed = gzip.compress(json.dumps(item).encode("utf-8"))

                s3.put_object(
                    Bucket="crypto-search-pipeline-iqtedar",
                    Key=s3_key,
                    Body=compressed
                )
                print(f"Uploaded new article: {title}")

          except Exception as e:
              import traceback
              print(f"[ERROR] Skipping article due to exception: {e}")
              traceback.print_exc()
              continue
        # if starting database, set to max pages, then adjust based on how often you batch and data load
        if( page_number == 1816):
           break
        page_number+=1
        print(f'page set to {page_number}')
       except Exception as e:
        #  Any errors, skip adding and continue adding artciles
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = f"An error occurred at {current_datetime}: {str(e)}"
        logging.error(error_message)
        page_number += 1
        print(f'page set to {page_number}')
        continue

if __name__ == "__main__":
    scrape_articles()