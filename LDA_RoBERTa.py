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
