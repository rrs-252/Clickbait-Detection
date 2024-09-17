import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim import corpora
from gensim.models import LdaMulticore
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from html_parser_preprocessor import HTMLParserPreprocessor  # Import our new class

# Define the path to your dataset
DATASET_PATH = "path/to/your/dataset"

# Initialize our HTML parser and preprocessor
parser = HTMLParserPreprocessor()

# Load metadata
metadata = pd.read_csv(os.path.join(DATASET_PATH, "articles_metadata.csv"))

# Process HTML files
texts = []
labels = []
for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing HTML files"):
    file_path = os.path.join(DATASET_PATH, "html_files", f"{row['domain']}_{row['category']}.html")
    processed_text = parser.parse_and_extract(file_path)
    texts.append(processed_text)
    labels.append(row['category'])

# LDA part
dictionary = corpora.Dictionary([text.split() for text in texts])
corpus = [dictionary.doc2bow(text.split()) for text in texts]

lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=10, workers=2)

def get_lda_features(text):
    bow = dictionary.doc2bow(text.split())
    return [topic_prob for topic, topic_prob in lda_model.get_document_topics(bow)]

# BERT part
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# Combine LDA and BERT features
X = np.array([np.concatenate([get_lda_features(text), get_bert_embedding(text)]) for text in tqdm(texts, desc="Extracting features")])
y = LabelEncoder().fit_transform(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch dataset and dataloader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the hybrid model
class LDABERTHybrid(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = X.shape[1]
num_classes = len(np.unique(y))
model = LDABERTHybrid(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total:.2f}%')

# Final evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f'Final Test Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'lda_bert_model.pth')
print("Model saved as lda_bert_model.pth")

# Function to predict category for new articles
def predict_category(model, html_source):
    model.eval()
    with torch.no_grad():
        # Process the HTML source
        processed_text = parser.parse_and_extract(html_source)
        
        # Extract features
        features = np.concatenate([get_lda_features(processed_text), get_bert_embedding(processed_text)])
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make prediction
        output = model(features)
        _, predicted = torch.max(output.data, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0]

# Example usage of the prediction function
new_article_url = "https://example.com/new_article"
predicted_category = predict_category(model, new_article_url)
print(f"Predicted category: {predicted_category}")
