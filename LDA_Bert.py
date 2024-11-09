import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim import corpora
from gensim.models import LdaMulticore
from transformers import BertTokenizer, BertModel
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

# BERT part
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

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
        bert_tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': bert_tokens['input_ids'].squeeze(),
            'attention_mask': bert_tokens['attention_mask'].squeeze(),
            'lda_features': torch.tensor(lda_features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DataLoader
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the LDA-BERT model
class LDABertClassifier(nn.Module):
    def __init__(self, bert_model, num_lda_features):
        super().__init__()
        self.bert_model = bert_model
        self.num_lda_features = num_lda_features
        self.classifier = nn.Linear(bert_model.config.hidden_size + num_lda_features, 2)

    def forward(self, input_ids, attention_mask, lda_features):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[1]
        combined_features = torch.cat((bert_output, lda_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# Initialize the model with correct number of LDA features
lda_bert_model = LDABertClassifier(bert_model, num_topics)  # Using num_topics instead of undefined lda_features

# Training hyperparameters
num_epochs = 10
learning_rate = 1e-5
batch_size = 8

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lda_bert_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    lda_bert_model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        lda_features = batch['lda_features']
        labels = batch['label']
        outputs = lda_bert_model(input_ids, attention_mask, lda_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}")

# Evaluation
lda_bert_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        lda_features = batch['lda_features']
        labels = batch['label']
        outputs = lda_bert_model(input_ids, attention_mask, lda_features)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

lda_bert_accuracy = accuracy_score(y_true, y_pred)
lda_bert_precision = precision_score(y_true, y_pred, average='weighted')
lda_bert_recall = recall_score(y_true, y_pred, average='weighted')
lda_bert_f1 = f1_score(y_true, y_pred, average='weighted')

print("LDA-BERT Model Performance:")
print(f"Accuracy: {lda_bert_accuracy:.4f}")
print(f"Precision: {lda_bert_precision:.4f}")
print(f"Recall: {lda_bert_recall:.4f}")
print(f"F1-Score: {lda_bert_f1:.4f}")

# Output verification
if lda_bert_accuracy > 0.8 and lda_bert_f1 > 0.8:
    print("Model training successful!")
else:
    print("Model training unsuccessful. Please check the model and data.")
    print("Accuracy:", lda_bert_accuracy)
    print("F1-Score:", lda_bert_f1)
