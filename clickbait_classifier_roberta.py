import sys
import torch
from html_parser_preprocessor import HTMLParserPreprocessor
from LDA_RoBERTa import load_saved_model
import torch.nn.functional as F

class ClickbaitClassifierRoBERTa:
    def __init__(self, model_dir='./saved_model_roberta'):
        self.html_parser = HTMLParserPreprocessor()
        print("Loading the model...")
        self.predictor, self.config = load_saved_model(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Model loaded successfully!")

    def process_input(self, input_text):
        try:
            return self.html_parser.parse_and_extract(input_text)
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return None

    def classify(self, processed_text):
        try:
            result = self.predictor.predict(processed_text)
            probabilities = result['probabilities']
            confidence = max(probabilities) * 100
            
            return {
                'classification': result['label'],
                'confidence': confidence
            }

        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return None

    def run_interactive(self):
        print("\nClickbait Classifier")
        print("-" * 20)
        print("Enter a URL or paste HTML content (type 'quit' to exit)")
        
        while True:
            try:
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
                        print("\nResults:")
                        print("-" * 8)
                        print(f"Classification: {result['classification'].upper()}")
                        print(f"Confidence: {result['confidence']:.2f}%")
                    else:
                        print("Classification failed. Please try again.")
                else:
                    print("Failed to process input. Please check your URL or HTML content.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                print("Please try again.")

def main():
    try:
        print("\nInitializing Clickbait Classifier...")
        classifier = ClickbaitClassifierRoBERTa()
        classifier.run_interactive()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("\nExiting program.")

if __name__ == "__main__":
    main()
