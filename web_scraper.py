import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urlparse
import time
import random

def get_domain(url):
    return urlparse(url).netloc

def scrape_article(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title (assuming it's in a h1 tag)
        title = soup.find('h1').text.strip() if soup.find('h1') else "No title found"
        
        # Extract content (assuming it's in p tags within a main or article tag)
        main_content = soup.find('main') or soup.find('article')
        if main_content:
            content = ' '.join([p.text for p in main_content.find_all('p')])
        else:
            content = ' '.join([p.text for p in soup.find_all('p')])
        
        return title, content
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None, None

def categorize_article(title, content):
    # This is a simple categorization. You might want to use a more sophisticated method.
    keywords = {
        'technology': ['tech', 'software', 'hardware', 'AI', 'robotics'],
        'politics': ['government', 'election', 'policy', 'law', 'president'],
        'sports': ['football', 'basketball', 'soccer', 'athlete', 'tournament'],
        'entertainment': ['movie', 'music', 'celebrity', 'film', 'TV show']
    }
    
    text = (title + ' ' + content).lower()
    for category, words in keywords.items():
        if any(word in text for word in words):
            return category
    return 'other'

def create_dataset(urls, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = []
    for url in urls:
        print(f"Scraping {url}")
        title, content = scrape_article(url)
        if title and content:
            category = categorize_article(title, content)
            domain = get_domain(url)
            data.append({
                'url': url,
                'domain': domain,
                'title': title,
                'content': content,
                'category': category
            })
        
        # Be polite: wait between requests
        time.sleep(random.uniform(1, 3))
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'articles_metadata.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")
    
    # Save individual HTML files
    html_dir = os.path.join(output_dir, 'html_files')
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    
    for _, row in df.iterrows():
        file_name = f"{row['domain']}_{row['category']}.html"
        file_path = os.path.join(html_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"<h1>{row['title']}</h1>\n<article>{row['content']}</article>")
    
    print(f"HTML files saved in {html_dir}")

# Example usage
urls = [
    'https://www.bbc.com/news/technology-60780142',
    'https://www.cnn.com/2023/05/02/politics/debt-ceiling-biden-mccarthy-meeting/index.html',
    'https://www.espn.com/nba/story/_/id/35545127/lebron-james-breaks-kareem-abdul-jabbar-all-scoring-record',
    'https://variety.com/2023/film/news/barbie-movie-trailer-margot-robbie-1235570019/'
]

create_dataset(urls, 'dataset')
