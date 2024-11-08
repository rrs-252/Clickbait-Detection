import torch
import os
from transformers import BertTokenizer, BertModel
from gensim import corpora
from gensim.models import LdaMulticore
import torch
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_Bert import LdaBertClassifier, dictionary, lda_model

class ClickbaitClassifier:
    def __init__(self):
        # Initialize HTML parser, BERT model, and tokenizer
        self.parser = HTMLParserPreprocessor()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = ['clickbait', 'not clickbait']  # Ensure these match your training labels
        self.lda_bert_model = LdaBertClassifier()

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

        # Step 2: Get LDA and BERT features
        lda_features = get_lda_features(processed_text)
        bert_tokens = self.tokenizer(processed_text,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors="pt")

        # Step 3: Get BERT embeddings
        with torch.no_grad():
            bert_output = self.bert_model(
                input_ids=bert_tokens['input_ids'],
                attention_mask=bert_tokens['attention_mask']
            )
            bert_embedding = bert_output.last_hidden_state.mean(dim=1)

        # Step 4: Combine LDA and BERT features
        combined_features = torch.cat(
            (torch.tensor(lda_features), bert_embedding.squeeze()),
            dim=0
        )

        # Step 5: Use the pre-trained LDA-BERT model to predict
        predicted_label = self.lda_bert_model.predict(combined_features)

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
