import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from LDA_Bert import load_saved_model as load_bert_model
from LDA_RoBERTa import load_saved_model as load_roberta_model
from html_parser_preprocessor import preprocess_text 

# Define the paths to the saved models
bert_model_dir = './saved_model'
roberta_model_dir = './saved_model_roberta'

# Define the paths to the special test dataset
clickbait_test_path = './test_data/test_clickbait.txt'
not_clickbait_test_path = './test_data/test_not_clickbait.txt'

# Load the saved models
print("Loading BERT model...")
bert_predictor, bert_config = load_bert_model(bert_model_dir)
print("BERT model loaded successfully.")

print("Loading RoBERTa model...")
roberta_predictor, roberta_config = load_roberta_model(roberta_model_dir)
print("RoBERTa model loaded successfully.")

# Load and preprocess the special test dataset
print("Loading and preprocessing special test dataset...")
special_test_texts = []
special_test_labels = []

with open(clickbait_test_path, 'r') as f:
    for line in f:
        cleaned_text = preprocess_text(line.strip())  # Preprocess each line
        special_test_texts.append(cleaned_text)
        special_test_labels.append('clickbait')

with open(not_clickbait_test_path, 'r') as f:
    for line in f:
        cleaned_text = preprocess_text(line.strip())  # Preprocess each line
        special_test_texts.append(cleaned_text)
        special_test_labels.append('not clickbait')

print("Special test dataset loaded and preprocessed successfully.")

# Evaluate the models on the preprocessed special test dataset
print("\nEvaluating models on the special test dataset:")

print("BERT Model:")
bert_predictions = [bert_predictor.predict(text)['label'] for text in special_test_texts]
bert_accuracy = accuracy_score(special_test_labels, bert_predictions)
bert_precision = precision_score(special_test_labels, bert_predictions, average='weighted')
bert_recall = recall_score(special_test_labels, bert_predictions, average='weighted')
bert_f1 = f1_score(special_test_labels, bert_predictions, average='weighted')
bert_confusion_matrix = confusion_matrix(special_test_labels, bert_predictions)
print(f"Accuracy: {bert_accuracy:.4f}")
print(f"Precision: {bert_precision:.4f}")
print(f"Recall: {bert_recall:.4f}")
print(f"F1-score: {bert_f1:.4f}")
print("Confusion Matrix:")
print(bert_confusion_matrix)

print("\nRoBERTa Model:")
roberta_predictions = [roberta_predictor.predict(text)['label'] for text in special_test_texts]
roberta_accuracy = accuracy_score(special_test_labels, roberta_predictions)
roberta_precision = precision_score(special_test_labels, roberta_predictions, average='weighted')
roberta_recall = recall_score(special_test_labels, roberta_predictions, average='weighted')
roberta_f1 = f1_score(special_test_labels, roberta_predictions, average='weighted')
roberta_confusion_matrix = confusion_matrix(special_test_labels, roberta_predictions)
print(f"Accuracy: {roberta_accuracy:.4f}")
print(f"Precision: {roberta_precision:.4f}")
print(f"Recall: {roberta_recall:.4f}")
print(f"F1-score: {roberta_f1:.4f}")
print("Confusion Matrix:")
print(roberta_confusion_matrix)

# Plot the confusion matrices
print("\nVisualizing Confusion Matrices:")

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

# Compare the models
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
