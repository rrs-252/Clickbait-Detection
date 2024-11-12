import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from typing import Union, IO

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class HTMLParserPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def parse_and_extract(self, source: Union[str, IO]) -> str:
        """
        Parse HTML from file content or URL and extract text content.
        
        Args:
            source: Can be one of:
                   - URL string starting with 'http'
                   - File object containing HTML content
                   - String containing HTML content
        
        Returns:
            Preprocessed text content ready for LDA-BERT processing
        """
        if isinstance(source, str):
            if source.startswith('http'):
                # It's a URL
                response = requests.get(source)
                html_content = response.text
            else:
                # It's a raw HTML string
                html_content = source
        else:
            # It's a file object
            html_content = source.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title
        title = soup.find('h1').text.strip() if soup.find('h1') else ""

        # Extract main content (adjust selectors based on typical structure of your target websites)
        content_tags = soup.find_all(['p', 'article', 'div', 'section'])
        content = ' '.join([tag.get_text(strip=True) for tag in content_tags])

        # Combine title and content
        full_text = f"{title} {content}"

        return self.preprocess(full_text)

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to the text."""
        text = self.lowercase(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_characters(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_html_tags(self, text: str) -> str:
        """Remove any remaining HTML tags."""
        return re.sub(r'<[^>]+>', '', text)

    def remove_special_characters(self, text: str) -> str:
        """Remove special characters and numbers."""
        # Keep only letters and spaces
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def tokenize(self, text: str) -> list:
        """Tokenize the text."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """Remove stop words."""
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens: list) -> list:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
