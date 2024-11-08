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
import torch.nn as nn
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
        texts.append(line.strip())  # each line is an article
        labels.append("clickbait")  # label as clickbait

# Load non-clickbait articles
with open(NOT_CLICKBAIT_PATH, 'r') as file:
    for line in file:
        texts.append(line.strip())  # each line is an article
        labels.append("not clickbait")  # label as not clickbait

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42)

# LDA part
dictionary = corpora.Dictionary([text.split() for text in train_texts])
corpus = [dictionary.doc2bow(text.split()) for text in train_texts]
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=10, workers=2)

def get_lda_features(text):
    bow = dictionary.doc2bow(text.split())
    return [topic_prob for topic, topic_prob in lda_model.get_document_topics(bow)]

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

# Model, training loop, etc. can follow here
# Export dictionary, lda_model, and TextDataset for use in other scripts
__all__ = ['dictionary', 'lda_model', 'TextDataset', 'get_lda_features']

# Any standalone execution code, such as model training or evaluation, can follow here if present.
# Define the LDA-RoBERTa model
class LDARoBERTaClassifier(nn.Module):
    def __init__(self, roberta_model, num_lda_features):
        super().__init__()
        self.roberta_model = roberta_model
        self.num_lda_features = num_lda_features
        self.classifier = nn.Linear(roberta_model.config.hidden_size + num_lda_features, 2)  # 2 classes (clickbait, not clickbait)

    def forward(self, input_ids, attention_mask, lda_features, pooled_output):
        combined_features = torch.cat((pooled_output, lda_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# Training hyperparameters
num_epochs = 10
learning_rate = 1e-5
batch_size = 8

# Initialize the model
lda_roberta_model = LDARoBERTaClassifier(roberta_model, len(lda_features[0]))

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
test_dataset = TextDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
