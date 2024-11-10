import sys
import torch
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_Bert import load_saved_model
import torch.nn.functional as F

class ClickbaitClassifier:
    def __init__(self, model_dir='./saved_model'):
        self.html_parser = HTMLParserPreprocessor()
        print("Loading the model...")
        self.model, self.tokenizer, self.lda_model, self.dictionary, \
            self.label_encoder, self.config = load_saved_model(model_dir)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def process_input(self, input_text):
        try:
            return self.html_parser.parse_and_extract(input_text)
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return None

    def classify(self, processed_text):
        try:
            with torch.no_grad():
                # Prepare LDA features
                bow = self.dictionary.doc2bow(processed_text.split())
                topic_probs = [0] * self.config['num_lda_topics']
                for topic, prob in self.lda_model.get_document_topics(bow):
                    topic_probs[topic] = prob

                # Prepare BERT features
                bert_tokens = self.tokenizer(
                    processed_text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config['max_length'],
                    return_tensors="pt"
                )

                input_ids = bert_tokens['input_ids'].to(self.device)
                attention_mask = bert_tokens['attention_mask'].to(self.device)
                lda_features = torch.tensor([topic_probs], dtype=torch.float).to(self.device)

                outputs = self.model(input_ids, attention_mask, lda_features)
                prediction = torch.argmax(outputs, dim=1)
                return self.label_encoder.inverse_transform(prediction.cpu().numpy())[0]

        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return None

    def run_interactive(self):
        print("\nClickbait Classifier")
        print("Enter a URL or paste HTML content (type 'quit' to exit)")
        
        while True:
            user_input = input("\nEnter URL or HTML content: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                print("Please provide valid input!")
                continue

            processed_text = self.process_input(user_input)
            
            if processed_text:
                result = self.classify(processed_text)
                if result:
                    print(f"Result: {result.upper()}")
                else:
                    print("Classification failed. Please try again.")
            else:
                print("Failed to process input. Please check your URL or HTML content.")

def main():
    try:
        classifier = ClickbaitClassifier()
        classifier.run_interactive()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("\nExiting program.")

if __name__ == "__main__":
    main()
