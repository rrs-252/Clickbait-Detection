import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class HTMLParserPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def parse_and_extract(self, source):
        """Parse HTML from file or URL and extract text content."""
        if source.startswith('http'):
            # It's a URL
            response = requests.get(source)
            html_content = response.text
        else:
            # It's a file path
            with open(source, 'r', encoding='utf-8') as file:
                html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title
        title = soup.find('h1').text.strip() if soup.find('h1') else ""

        # Extract main content (adjust selectors based on typical structure of your target websites)
        content_tags = soup.find_all(['p', 'article', 'div', 'section'])
        content = ' '.join([tag.get_text(strip=True) for tag in content_tags])

        # Combine title and content
        full_text = f"{title} {content}"

        return self.preprocess(full_text)

    def preprocess(self, text):
        """Apply all preprocessing steps to the text."""
        text = self.lowercase(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_characters(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)  # Join tokens back into a string for LDA

    def lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def remove_html_tags(self, text):
        """Remove any remaining HTML tags."""
        return re.sub(r'<[^>]+>', '', text)

    def remove_special_characters(self, text):
        """Remove special characters and numbers."""
        # Keep only letters and spaces
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def tokenize(self, text):
        """Tokenize the text."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove stop words."""
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens):
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

# Example usage
parser = HTMLParserPreprocessor()

# Process a URL
url = "https://example.com/article"
processed_text_url = parser.parse_and_extract(url)
print("Processed text from URL:", processed_text_url[:100])  # Print first 100 characters

# Process a local HTML file
file_path = "path/to/local/file.html"
processed_text_file = parser.parse_and_extract(file_path)
print("Processed text from file:", processed_text_file[:100])  # Print first 100 characters
