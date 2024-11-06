import torch
import os
from transformers import BertTokenizer, BertModel
from gensim import corpora
from gensim.models import LdaMulticore
import torch
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_Bert import dictionary, lda_model, get_lda_features

class ClickbaitClassifier:
    def __init__(self):
        # Initialize HTML parser, BERT model, and tokenizer
        self.parser = HTMLParserPreprocessor()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
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

        # Load a pre-trained classifier model here and predict with `combined_features`
        # Example: classifier_output = classifier_model(combined_features)
        # predicted_label = torch.argmax(classifier_output).item()
        predicted_label = 0  # Replace this with the actual predicted label

        # Step 5: Decode label and output result
        prediction_label = self.label_encoder.inverse_transform([predicted_label])[0]
        return prediction_label

def main():
    # Initialize the classifier
    classifier = ClickbaitClassifier()

    # Example usage with different input types:

    # 1. From URL
    url = "https://example.com/article"
    result_url = classifier.predict_clickbait(url)
    print(f"URL article classification: {result_url}")

    # 2. From file object
    with open('article.html', 'r', encoding='utf-8') as file:
        result_file = classifier.predict_clickbait(file)
    print(f"File article classification: {result_file}")

    # 3. From HTML string
    html_string = "<html><body><h1>Article Title</h1><p>Article content...</p></body></html>"
    result_string = classifier.predict_clickbait(html_string)
    print(f"HTML string classification: {result_string}")

if __name__ == "__main__":
    main()
