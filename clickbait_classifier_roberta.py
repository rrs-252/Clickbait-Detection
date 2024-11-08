import torch
import os
from transformers import RobertaTokenizer, RobertaModel
from gensim import corpora
from gensim.models import LdaMulticore
import torch
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_RoBERTa import LdaRobertaClassifier, dictionary, lda_model

class ClickbaitClassifier:
    def __init__(self):
        # Initialize HTML parser, RoBERTa model, and tokenizer
        self.parser = HTMLParserPreprocessor()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = ['clickbait', 'not clickbait']  # Ensure these match your training labels
        self.lda_roberta_model = LdaRobertaClassifier()

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
            roberta_output = self.roberta_model(
                input_ids=roberta_tokens['input_ids'],
                attention_mask=roberta_tokens['attention_mask']
            )
            roberta_embedding = roberta_output.last_hidden_state.mean(dim=1)

        # Step 4: Combine LDA and RoBERTa features
        combined_features = torch.cat(
            (torch.tensor(lda_features), roberta_embedding.squeeze()),
            dim=0
        )

        # Step 5: Use the pre-trained LDA-RoBERTa model to predict
        predicted_label = self.lda_roberta_model.predict(combined_features)

        # Step 6: Decode label and output result
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
