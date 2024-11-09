import torch
import os
from transformers import BertTokenizer, BertModel
from gensim import corpora
from gensim.models import LdaMulticore
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_Bert import LdaBertClassifier, dictionary, lda_model

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
        
        # Load the pre-trained LDA-BERT model
        self.lda_bert_model = LdaBertClassifier()
        self.load_trained_model(model_path)

    def load_trained_model(self, model_path):
        """Load the trained model from the specified path"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.lda_bert_model.load_state_dict(checkpoint['model_state_dict'])
            self.lda_bert_model.to(self.device)
            self.lda_bert_model.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model from {model_path}: {str(e)}")

    def get_lda_features(self, text):
        """Extract LDA features from text"""
        # Tokenize and get document-term matrix
        bow = dictionary.doc2bow(text.split())
        # Get topic distribution
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        # Convert to dense vector
        lda_features = [0] * lda_model.num_topics
        for topic_id, prob in topic_dist:
            lda_features[topic_id] = prob
        return torch.tensor(lda_features, dtype=torch.float32, device=self.device)

    @torch.no_grad()  # Disable gradient calculation for inference
    def predict_clickbait(self, html_content):
        """
        Predict whether content is clickbait or not using GPU acceleration.
        Args:
            html_content: Can be one of:
                       - URL string starting with 'http'
                       - File object containing HTML content
                       - String containing HTML content
        Returns:
            str: 'clickbait' or 'not clickbait'
        """
        try:
            # Step 1: Parse HTML content
            processed_text = self.parser.parse_and_extract(html_content)
            
            # Step 2: Get LDA features
            lda_features = self.get_lda_features(processed_text)
            
            # Step 3: Get BERT features
            bert_tokens = self.tokenizer(
                processed_text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Step 4: Get BERT embeddings
            bert_output = self.bert_model(
                input_ids=bert_tokens['input_ids'],
                attention_mask=bert_tokens['attention_mask']
            )
            bert_embedding = bert_output.last_hidden_state.mean(dim=1)
            
            # Step 5: Combine LDA and BERT features
            combined_features = torch.cat(
                (lda_features, bert_embedding.squeeze()),
                dim=0
            )
            
            # Step 6: Use the pre-trained model to predict
            self.lda_bert_model.eval()
            logits = self.lda_bert_model(combined_features.unsqueeze(0))
            predicted_label = torch.argmax(logits, dim=1).item()
            
            return self.label_encoder.inverse_transform([predicted_label])[0]
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Error in prediction"
        finally:
            # Clear GPU memory
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

    # Get user input
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
            
        # Clear GPU memory after each prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
