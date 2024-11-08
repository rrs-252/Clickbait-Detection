import torch
import os
from transformers import RobertaTokenizer, RobertaModel
from gensim import corpora
from gensim.models import LdaMulticore
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_RoBERTa import dictionary, lda_model, get_lda_features, TextDataset

class ClickbaitClassifier:
    def __init__(self):
        # Initialize HTML parser, RoBERTa model, and tokenizer
        self.parser = HTMLParserPreprocessor()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = ['clickbait', 'not clickbait']  # Ensure these match your training labels

    def predict_clickbait(self, html_content):
        """
        Predict whether content is clickbait or not.

        Args:
            html_content: Can be one of:
                       - URL string starting with 'http'
                       - File object containing HTML content
                       - String containing HTML content

        Returns:
            str: 'clickbait' or 'not clickbait'
        """
        # Step 1: Parse HTML content using the updated parser
        try:
            processed_text = self.parser.parse_and_extract(html_content)
        except Exception as e:
            raise ValueError(f"Error processing HTML content: {str(e)}")

        # Step 2: Get LDA and RoBERTa features
        lda_features = get_lda_features(processed_text)
        roberta_tokens = self.tokenizer(processed_text,
                                       padding='max_length',
                                       truncation=True,
                                       return_tensors="pt")

        # Step 3: Get RoBERTa embeddings
        with torch.no_grad():
            roberta_output = self.roberta_model(**roberta_tokens)
            roberta_embedding = roberta_output.pooler_output.squeeze()

        # Step 4: Combine LDA and RoBERTa features
        combined_features = torch.cat(
            (torch.tensor(lda_features), roberta_embedding),
            dim=0
        )

        # Load a pre-trained classifier model here and predict with `combined_features`
        # Example: classifier_output = classifier_model(combined_features)
        predicted_label = 0  # Replace this with the actual predicted label (an integer)

        # Step 5: Decode label and output result
        if predicted_label == 0:
            return "not clickbait"
        else:
            return "clickbait"

def main():
    # Initialize the classifier
    classifier = ClickbaitClassifier()

    # Get user input
    while True:
        user_input = input("Enter a website URL, HTML file path, or HTML content (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        if user_input.startswith('http'):
            result = classifier.predict_clickbait(user_input)
            print(f"URL article classification: {result}")
        elif os.path.isfile(user_input):
            with open(user_input, 'r', encoding='utf-8') as file:
                result = classifier.predict_clickbait(file)
            print(f"File article classification: {result}")
        else:
            result = classifier.predict_clickbait(user_input)
            print(f"HTML string classification: {result}")

if __name__ == "__main__":
    main()
