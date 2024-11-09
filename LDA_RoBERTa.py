import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim import corpora
from gensim.models import LdaMulticore
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths to the datasets
CLICKBAIT_PATH = "articlesClickbait.txt"
NOT_CLICKBAIT_PATH = "articlesNotClickbait.txt"

# Load data from files
texts = []
labels = []

# Load clickbait articles
with open(CLICKBAIT_PATH, 'r') as file:
    for line in file:
        texts.append(line.strip())
        labels.append("clickbait")

# Load non-clickbait articles
with open(NOT_CLICKBAIT_PATH, 'r') as file:
    for line in file:
        texts.append(line.strip())
        labels.append("not clickbait")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42)

# LDA part
dictionary = corpora.Dictionary([text.split() for text in train_texts])
corpus = [dictionary.doc2bow(text.split()) for text in train_texts]
num_topics = 10  # Define number of topics
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=2)

def get_lda_features(text):
    bow = dictionary.doc2bow(text.split())
    # Initialize a zero vector for all topics
    topic_probs = [0] * num_topics
    # Fill in the probabilities for the topics that appear
    for topic, prob in lda_model.get_document_topics(bow):
        topic_probs[topic] = prob
    return topic_probs

# RoBERTa part
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        lda_features = get_lda_features(text)
        roberta_tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        with torch.no_grad():  # Add this to prevent gradient computation during data loading
            roberta_outputs = roberta_model(**roberta_tokens)
        return {
            'input_ids': roberta_tokens['input_ids'].squeeze(),
            'attention_mask': roberta_tokens['attention_mask'].squeeze(),
            'lda_features': torch.tensor(lda_features, dtype=torch.float),
            'pooled_output': roberta_outputs.pooler_output.squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DataLoader
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the LDA-RoBERTa model
class LDARoBERTaClassifier(nn.Module):
    def __init__(self, roberta_model, num_lda_features):
        super().__init__()
        self.roberta_model = roberta_model
        self.num_lda_features = num_lda_features
        self.classifier = nn.Linear(roberta_model.config.hidden_size + num_lda_features, 2)

    def forward(self, input_ids, attention_mask, lda_features, pooled_output):
        combined_features = torch.cat((pooled_output, lda_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# Initialize the model with correct number of LDA features
lda_roberta_model = LDARoBERTaClassifier(roberta_model, num_topics)

# Training hyperparameters
num_epochs = 10
learning_rate = 1e-5
batch_size = 8

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lda_roberta_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    lda_roberta_model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        lda_features = batch['lda_features']
        pooled_output = batch['pooled_output']
        labels = batch['label']
        outputs = lda_roberta_model(input_ids, attention_mask, lda_features, pooled_output)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}")

# Evaluation
lda_roberta_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        lda_features = batch['lda_features']
        pooled_output = batch['pooled_output']
        labels = batch['label']
        outputs = lda_roberta_model(input_ids, attention_mask, lda_features, pooled_output)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

lda_roberta_accuracy = accuracy_score(y_true, y_pred)
lda_roberta_precision = precision_score(y_true, y_pred, average='weighted')
lda_roberta_recall = recall_score(y_true, y_pred, average='weighted')
lda_roberta_f1 = f1_score(y_true, y_pred, average='weighted')

print("LDA-RoBERTa Model Performance:")
print(f"Accuracy: {lda_roberta_accuracy:.4f}")
print(f"Precision: {lda_roberta_precision:.4f}")
print(f"Recall: {lda_roberta_recall:.4f}")
print(f"F1-Score: {lda_roberta_f1:.4f}")

# Output verification
if lda_roberta_accuracy > 0.8 and lda_roberta_f1 > 0.8:
    print("Model training successful!")
else:
    print("Model training unsuccessful. Please check the model and data.")
    print("Accuracy:", lda_roberta_accuracy)
    print("F1-Score:", lda_roberta_f1)
