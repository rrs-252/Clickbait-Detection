import torch
import os
from transformers import BertTokenizer, BertModel
from gensim import corpora
from gensim.models import LdaMulticore
import torch
from sklearn.preprocessing import LabelEncoder
from html_parser_preprocessor import HTMLParserPreprocessor
from Modified_LDA_Bert import TextDataset, get_lda_features

# Initialize the HTML parser
parser = HTMLParserPreprocessor()

# Load pre-trained BERT and LDA models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Placeholder for loading LDA model (assuming it's saved or loaded with same dictionary from training)
dictionary = None  # Replace with the actual dictionary loaded from your LDA training
lda_model = None   # Replace with your actual LDA model (assuming saved and reloaded here)

# Set up label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = ['clickbait', 'not clickbait']  # Ensure these match your training labels

# Prediction Function
def predict_clickbait(html_content):
    # Step 1: Parse HTML content
    processed_text = parser.parse_and_extract(html_content)
    
    # Step 2: Get LDA and BERT features
    lda_features = get_lda_features(processed_text)
    bert_tokens = tokenizer(processed_text, padding='max_length', truncation=True, return_tensors="pt")

    # Step 3: Get BERT embeddings
    with torch.no_grad():
        bert_output = bert_model(input_ids=bert_tokens['input_ids'], attention_mask=bert_tokens['attention_mask'])
        bert_embedding = bert_output.last_hidden_state.mean(dim=1)  # Pool the BERT output

    # Step 4: Combine LDA and BERT features (Example: simple concatenation)
    combined_features = torch.cat((torch.tensor(lda_features), bert_embedding.squeeze()), dim=0)
    
    # Load a pre-trained classifier model here and predict with `combined_features`
    # Example: classifier_output = classifier_model(combined_features)
    # predicted_label = torch.argmax(classifier_output).item()
    
    # Step 5: Decode label and output result
    prediction_label = label_encoder.inverse_transform([predicted_label])[0]  # Assuming binary classification
    return prediction_label

# Main Function
def main():
    # Specify HTML file or URL
    html_content = "path/to/html_file.html"  # Update this to a file path or load HTML from URL
    
    # Predict if the content is clickbait or not
    result = predict_clickbait(html_content)
    print(f"The article is predicted as: {result}")

if __name__ == "__main__":
    main()
