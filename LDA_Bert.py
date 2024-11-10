# First, add these at the start of your notebook
import os
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim import corpora
from gensim.models import LdaMulticore
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# Memory management functions
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
def print_gpu_memory():
    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB')
    print(f'GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Modified data loading with chunking
def load_data_in_chunks(file_path, label, chunk_size=1000):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                texts.extend(chunk)
                labels.extend([label] * len(chunk))
                chunk = []
                clear_memory()
        if chunk:  # Don't forget the last chunk
            texts.extend(chunk)
            labels.extend([label] * len(chunk))
    return texts, labels
    
# Paths to the datasets
CLICKBAIT_PATH = "/train_data/train_Clickbait.txt"
NOT_CLICKBAIT_PATH = "/train_data/train_not_Clickbait.txt"

# Load data with chunking
texts = []
labels = []

chunk_texts, chunk_labels = load_data_in_chunks(CLICKBAIT_PATH, "clickbait")
texts.extend(chunk_texts)
labels.extend(chunk_labels)
clear_memory()

chunk_texts, chunk_labels = load_data_in_chunks(NOT_CLICKBAIT_PATH, "not clickbait")
texts.extend(chunk_texts)
labels.extend(chunk_labels)
clear_memory()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42)

# Free memory
del texts, labels, encoded_labels
clear_memory()

# Modified LDA part with memory efficiency
dictionary = corpora.Dictionary([text.split() for text in train_texts])
corpus = [dictionary.doc2bow(text.split()) for text in train_texts]
num_topics = 10
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, 
                        num_topics=num_topics, workers=2,
                        batch=True)  # Enable batch processing

# Free memory
del corpus
clear_memory()

# Modified Dataset class with memory optimization
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Reduced max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Get LDA features
        bow = dictionary.doc2bow(text.split())
        topic_probs = [0] * num_topics
        for topic, prob in lda_model.get_document_topics(bow):
            topic_probs[topic] = prob
            
        # Get BERT features
        bert_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': bert_tokens['input_ids'].squeeze(),
            'attention_mask': bert_tokens['attention_mask'].squeeze(),
            'lda_features': torch.tensor(topic_probs, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Modified model with gradient checkpointing
class LDABertClassifier(nn.Module):
    def __init__(self, bert_model, num_lda_features):
        super().__init__()
        self.bert_model = bert_model
        self.bert_model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        self.num_lda_features = num_lda_features
        self.classifier = nn.Linear(bert_model.config.hidden_size + num_lda_features, 2)

    def forward(self, input_ids, attention_mask, lda_features):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[1]
        combined_features = torch.cat((bert_output, lda_features), dim=1)
        return self.classifier(combined_features)

# Initialize model and move to GPU
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
lda_bert_model = LDABertClassifier(bert_model, num_topics).to(device)

# Create datasets with smaller batch size
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

# Smaller batch size and gradient accumulation
batch_size = 4
accumulation_steps = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training setup
num_epochs = 10
learning_rate = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(lda_bert_model.parameters(), lr=learning_rate)

# Modified training loop with gradient accumulation and memory management
for epoch in range(num_epochs):
    lda_bert_model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_loader)):
        # Move batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lda_features = batch['lda_features'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = lda_bert_model(input_ids, attention_mask, lda_features)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps  # Normalize loss
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            clear_memory()
        
        train_loss += loss.item() * accumulation_steps
        
        # Print memory usage every 100 batches
        if i % 100 == 0:
            print_gpu_memory()
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': lda_bert_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'checkpoint_epoch_{epoch}.pt')
    
    clear_memory()

# Evaluation with memory management
lda_bert_model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lda_features = batch['lda_features'].to(device)
        labels = batch['label'].to(device)
        
        outputs = lda_bert_model(input_ids, attention_mask, lda_features)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())
        
        clear_memory()

# Calculate metrics
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, average='weighted'),
    'recall': recall_score(y_true, y_pred, average='weighted'),
    'f1': f1_score(y_true, y_pred, average='weighted')
}

print("\nFinal Results:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

if metrics['accuracy'] > 0.8 and metrics['f1'] > 0.8:
    print("Model training successful!")
else:
    print("Model training needs improvement.")

# Create a directory for the model if it doesn't exist
os.makedirs('saved_model', exist_ok=True)

# Save the complete model
model_save_path = 'saved_model/lda_bert_model'
config_save_path = 'saved_model/model_config.json'

# Save the model state and architecture
torch.save({
    'model_state_dict': lda_bert_model.state_dict(),
    'model_config': {
        'bert_model_name': 'bert-base-uncased',
        'num_lda_topics': num_topics,
        'max_length': train_dataset.max_length
    }
}, f'{model_save_path}.pt')

# Save the LDA model
lda_model.save(f'{model_save_path}_lda.model')

# Save the dictionary
dictionary.save(f'{model_save_path}_dictionary.dict')

# Save the label encoder
with open(f'{model_save_path}_label_encoder.npy', 'wb') as f:
    np.save(f, label_encoder.classes_)

# Save additional configuration and metadata
config = {
    'model_performance': metrics,
    'training_params': {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'accumulation_steps': accumulation_steps
    },
    'tokenizer_name': 'bert-base-uncased',
    'max_length': train_dataset.max_length,
    'num_lda_topics': num_topics
}

with open(config_save_path, 'w') as f:
    json.dump(config, f, indent=4)

print("\nModel saved successfully!")
print(f"Model files saved in: {os.path.abspath('saved_model')}")

# Function to load the saved model (for future use)
def load_saved_model(model_dir='saved_model'):
    # Load configuration
    with open(f'{model_dir}/model_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Load BERT and initialize combined model
    bert_model = BertModel.from_pretrained(config['tokenizer_name'])
    model = LDABertClassifier(bert_model, config['num_lda_topics'])
    
    # Load model state
    checkpoint = torch.load(f'{model_dir}/lda_bert_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load LDA model
    lda_model = LdaMulticore.load(f'{model_dir}/lda_bert_model_lda.model')
    
    # Load dictionary
    dictionary = corpora.Dictionary.load(f'{model_dir}/lda_bert_model_dictionary.dict')
    
    # Load label encoder classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(f'{model_dir}/lda_bert_model_label_encoder.npy', allow_pickle=True)
    
    return model, tokenizer, lda_model, dictionary, label_encoder, config

print("\nTo load the saved model later, use the load_saved_model() function.")
