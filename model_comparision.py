import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from LDA_Bert import load_saved_model as load_bert_model
from LDA_RoBERTa import load_saved_model as load_roberta_model
from html_parser_preprocessor import HTMLParserPreprocessor
from transformers import BertTokenizer

# Create an instance of the HTMLParserPreprocessor class
preprocessor = HTMLParserPreprocessor()

# Define the paths to the saved models
bert_model_dir = './saved_model'
roberta_model_dir = './saved_model_roberta'

# Define the paths to the special test dataset
clickbait_test_path = './test_data/test_clickbait.txt'
not_clickbait_test_path = './test_data/test_not_clickbait.txt'

# Load the saved BERT and RoBERTa models
print("Loading BERT model...")
bert_predictor = load_bert_model(bert_model_dir)
print("BERT model loaded successfully.")

print("Loading RoBERTa model...")
roberta_predictor, roberta_config = load_roberta_model(roberta_model_dir)
print("RoBERTa model loaded successfully.")

# Load the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load and preprocess the special test dataset
print("Loading and preprocessing special test dataset...")
special_test_texts = []
special_test_labels = []

with open(clickbait_test_path, 'r') as f:
    for line in f:
        cleaned_text = preprocessor.preprocess(line.strip())
        special_test_texts.append(cleaned_text)
        special_test_labels.append('clickbait')

with open(not_clickbait_test_path, 'r') as f:
    for line in f:
        cleaned_text = preprocessor.preprocess(line.strip())
        special_test_texts.append(cleaned_text)
        special_test_labels.append('not clickbait')

print("Special test dataset loaded and preprocessed successfully.")

# Evaluate the models on the preprocessed special test dataset
print("\nEvaluating models on the special test dataset:")

# BERT Model Predictions
print("BERT Model:")
bert_predictions = []
for text in special_test_texts:
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = bert_predictor(**inputs)
    predicted_label = outputs.logits.argmax(dim=-1).item()  # Adjust this as necessary for your BERT model's output
    bert_predictions.append('clickbait' if predicted_label == 1 else 'not clickbait')

# RoBERTa Model Predictions
print("RoBERTa Model:")
roberta_predictions = [roberta_predictor.predict(text)['label'] for text in special_test_texts]

# Calculate metrics for BERT
bert_accuracy = accuracy_score(special_test_labels, bert_predictions)
bert_precision = precision_score(special_test_labels, bert_predictions, average='weighted')
bert_recall = recall_score(special_test_labels, bert_predictions, average='weighted')
bert_f1 = f1_score(special_test_labels, bert_predictions, average='weighted')
bert_confusion_matrix = confusion_matrix(special_test_labels, bert_predictions)

# Print BERT metrics
print(f"BERT Accuracy: {bert_accuracy:.4f}")
print(f"BERT Precision: {bert_precision:.4f}")
print(f"BERT Recall: {bert_recall:.4f}")
print(f"BERT F1-score: {bert_f1:.4f}")
print("BERT Confusion Matrix:")
print(bert_confusion_matrix)

# Calculate metrics for RoBERTa
roberta_accuracy = accuracy_score(special_test_labels, roberta_predictions)
roberta_precision = precision_score(special_test_labels, roberta_predictions, average='weighted')
roberta_recall = recall_score(special_test_labels, roberta_predictions, average='weighted')
roberta_f1 = f1_score(special_test_labels, roberta_predictions, average='weighted')
roberta_confusion_matrix = confusion_matrix(special_test_labels, roberta_predictions)

# Print RoBERTa metrics
print(f"RoBERTa Accuracy: {roberta_accuracy:.4f}")
print(f"RoBERTa Precision: {roberta_precision:.4f}")
print(f"RoBERTa Recall: {roberta_recall:.4f}")
print(f"RoBERTa F1-score: {roberta_f1:.4f}")
print("RoBERTa Confusion Matrix:")
print(roberta_confusion_matrix)

# Plot the confusion matrices
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
bert_cm_display = ConfusionMatrixDisplay(confusion_matrix=bert_confusion_matrix, display_labels=['clickbait', 'not clickbait'])
bert_cm_display.plot()
plt.title("BERT Model Confusion Matrix")

plt.subplot(1, 2, 2)
roberta_cm_display = ConfusionMatrixDisplay(confusion_matrix=roberta_confusion_matrix, display_labels=['clickbait', 'not clickbait'])
roberta_cm_display.plot()
plt.title("RoBERTa Model Confusion Matrix")

plt.suptitle("Comparison of Confusion Matrices")
plt.show()

# Plot performance metrics comparison
plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.bar(['BERT', 'RoBERTa'], [bert_accuracy, roberta_accuracy])
plt.title('Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')

# F1-score
plt.subplot(1, 2, 2)
plt.bar(['BERT', 'RoBERTa'], [bert_f1, roberta_f1])
plt.title('F1-score Comparison')
plt.xlabel('Model')
plt.ylabel('F1-score')

plt.suptitle("Performance Metrics Comparison")
plt.tight_layout()
plt.show()

# Model Comparison
print("\nModel Comparison:")
print(f"BERT Model Accuracy: {bert_accuracy:.4f}")
print(f"RoBERTa Model Accuracy: {roberta_accuracy:.4f}")
if bert_accuracy > roberta_accuracy:
    print("BERT model performs better on the special test dataset.")
elif bert_accuracy < roberta_accuracy:
    print("RoBERTa model performs better on the special test dataset.")
else:
    print("Both models perform equally well on the special test dataset.")

print(f"\nBERT Model F1-score: {bert_f1:.4f}")
print(f"RoBERTa Model F1-score: {roberta_f1:.4f}")
if bert_f1 > roberta_f1:
    print("BERT model has a higher F1-score on the special test dataset.")
elif bert_f1 < roberta_f1:
    print("RoBERTa model has a higher F1-score on the special test dataset.")
else:
    print("Both models have equal F1-scores on the special test dataset.")
