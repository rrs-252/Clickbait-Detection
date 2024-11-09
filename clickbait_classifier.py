import torch
import os
from transformers import BertTokenizer, BertModel
from gensim import corpora
from gensim.models import LdaMulticore
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor

class ClickbaitClassifier:
    def __init__(self, model_path='/content/LDA-Bert/saved_model'):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize HTML parser
        self.parser = HTMLParserPreprocessor()
        
        # Initialize BERT on GPU
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = ['clickbait', 'not clickbait']
        
        # Load the dictionary and LDA model from saved files
        self.dictionary = corpora.Dictionary.load(os.path.join(model_path, 'dictionary.dict'))
        self.lda_model = LdaMulticore.load(os.path.join(model_path, 'lda_model.model'))
        
        # Load the pretrained classifier
        self.load_trained_classifier(model_path)

    def load_trained_classifier(self, model_path):
        """Load the pretrained classifier from the saved model"""
        try:
            # Load model architecture and weights
            model_file = os.path.join(model_path, 'clickbait_classifier.pth')
            self.classifier = torch.load(model_file, map_location=self.device)
            self.classifier.eval()  # Set to evaluation mode
            print(f"Successfully loaded classifier from {model_file}")
        except Exception as e:
            raise ValueError(f"Error loading classifier from {model_path}: {str(e)}")

    def get_lda_features(self, text):
        """Extract LDA features from text"""
        # Tokenize and get document-term matrix
        bow = self.dictionary.doc2bow(text.split())
        # Get topic distribution
        topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=0.0)
        # Convert to dense vector
        lda_features = [0] * self.lda_model.num_topics
        for topic_id, prob in topic_dist:
            lda_features[topic_id] = prob
        return torch.tensor(lda_features, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_clickbait(self, html_content):
        """
        Predict whether content is clickbait or not using the pretrained model.
        """
        try:
            # Parse HTML content
            processed_text = self.parser.parse_and_extract(html_content)
            
            # Get LDA features
            lda_features = self.get_lda_features(processed_text)
            
            # Get BERT features
            bert_tokens = self.tokenizer(
                processed_text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            bert_output = self.bert_model(
                input_ids=bert_tokens['input_ids'],
                attention_mask=bert_tokens['attention_mask']
            )
            bert_embedding = bert_output.last_hidden_state.mean(dim=1)
            
            # Combine features
            combined_features = torch.cat(
                (lda_features, bert_embedding.squeeze()),
                dim=0
            ).unsqueeze(0)  # Add batch dimension
            
            # Get prediction from pretrained classifier
            logits = self.classifier(combined_features)
            predicted_label = torch.argmax(logits, dim=1).item()
            
            return self.label_encoder.inverse_transform([predicted_label])[0]
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Error in prediction"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main():
    # Initialize the classifier
    try:
        classifier = ClickbaitClassifier()
        print("Classifier initialized successfully")
    except Exception as e:
        print(f"Error initializing classifier: {str(e)}")
        return

    while True:
        try:
            user_input = input("\nEnter a website URL, HTML file path, or HTML content (type 'quit' to exit): ")
            
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
                
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
