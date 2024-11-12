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
import glob

# Import the HTMLParserPreprocessor
from html_parser_preprocessor import HTMLParserPreprocessor

# Memory management functions
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
def print_gpu_memory():
    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB')
    print(f'GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB')

# Modified data loading with chunking and HTML parsing
def load_data_in_chunks(file_path, label, chunk_size=1000):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            # Use the HTMLParserPreprocessor to extract text from the HTML
            html_preprocessor = HTMLParserPreprocessor()
            text = html_preprocessor.parse_and_extract(line.strip())
            chunk.append(text)
            if len(chunk) >= chunk_size:
                texts.extend(chunk)
                labels.extend([label] * len(chunk))
                chunk = []
                clear_memory()
        if chunk:  # Don't forget the last chunk
            texts.extend(chunk)
            labels.extend([label] * len(chunk))
    return texts, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, dictionary, lda_model, num_topics, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.lda_model = lda_model
        self.num_topics = num_topics
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Get LDA features
        bow = self.dictionary.doc2bow(text.split())
        topic_probs = [0] * self.num_topics
        for topic, prob in self.lda_model.get_document_topics(bow):
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

class LDABertClassifier(nn.Module):
    def __init__(self, bert_model, num_lda_features):
        super().__init__()
        self.bert_model = bert_model
        self.bert_model.gradient_checkpointing_enable()
        self.num_lda_features = num_lda_features
        self.classifier = nn.Linear(bert_model.config.hidden_size + num_lda_features, 2)

    def forward(self, input_ids, attention_mask, lda_features):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[1]
        combined_features = torch.cat((bert_output, lda_features), dim=1)
        return self.classifier(combined_features)

def load_and_prepare_data(clickbait_path, not_clickbait_path):
    # Load data with chunking and HTML parsing
    texts = []
    labels = []

    chunk_texts, chunk_labels = load_data_in_chunks(clickbait_path, "clickbait")
    texts.extend(chunk_texts)
    labels.extend(chunk_labels)
    clear_memory()

    chunk_texts, chunk_labels = load_data_in_chunks(not_clickbait_path, "not clickbait")
    texts.extend(chunk_texts)
    labels.extend(chunk_labels)
    clear_memory()

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42)

    return train_texts, test_texts, train_labels, test_labels, label_encoder

def train_model(train_loader, test_loader, model, device, num_epochs, learning_rate, 
                accumulation_steps, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader)):
            # Move batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lda_features = batch['lda_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, lda_features)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                clear_memory()
            
            train_loss += loss.item() * accumulation_steps
            
            if i % 100 == 0:
                print_gpu_memory()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pt')
        
        clear_memory()
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lda_features = batch['lda_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, lda_features)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
            
            clear_memory()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def save_model(model, tokenizer, lda_model, dictionary, label_encoder, config, metrics, 
               save_dir='saved_model'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model state and architecture
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config
    }, f'{save_dir}/lda_bert_model.pt')

    # Save the LDA model
    lda_model.save(f'{save_dir}/lda_bert_model_lda.model')

    # Save the dictionary
    dictionary.save(f'{save_dir}/lda_bert_model_dictionary.dict')

    # Save the label encoder
    with open(f'{save_dir}/lda_bert_model_label_encoder.npy', 'wb') as f:
        np.save(f, label_encoder.classes_)

    # Save configuration and metadata
    config.update({'model_performance': metrics})
    with open(f'{save_dir}/model_config.json', 'w') as f:
        json.dump(config, f, indent=4)

def load_saved_model(model_dir='./saved_model', device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Define file paths
    config_path = f'{model_dir}/model_config.json'
    model_path = f'{model_dir}/lda_bert_model.pt'
    lda_path = f'{model_dir}/lda_bert_model_lda.model'
    dict_path = f'{model_dir}/lda_bert_model_dictionary.dict'
    label_encoder_path = f'{model_dir}/lda_bert_model_label_encoder.npy'
    
    # Check if all required files exist
    required_files = [config_path, model_path, lda_path, dict_path, label_encoder_path]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        else:
            print(f"Found file: {path}")

    # Load configuration
    print("Loading model configuration...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Load BERT and initialize combined model
    print("Initializing BERT model...")
    bert_model = BertModel.from_pretrained(config['tokenizer_name'])
    model = LDABertClassifier(bert_model, config['num_lda_topics']).to(device)
    
    # Load model state
    try:
        print("Loading model state...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Ensure the model is in evaluation mode
        print("Model state loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model state: {str(e)}")

    # Load LDA model
    try:
        print("Loading LDA model...")
        lda_model = LdaMulticore.load(lda_path)
        print("LDA model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading LDA model: {str(e)}")

    # Load dictionary
    try:
        print("Loading dictionary...")
        dictionary = corpora.Dictionary.load(dict_path)
        print("Dictionary loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading dictionary: {str(e)}")

    # Load label encoder classes
    try:
        print("Loading label encoder...")
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
        print("Label encoder loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading label encoder: {str(e)}")
    
    # Create a class to handle predictions
    class ModelPredictor:
        def __init__(self, model, tokenizer, lda_model, dictionary, label_encoder, device, max_length=256):
            self.model = model
            self.tokenizer = tokenizer
            self.lda_model = lda_model
            self.dictionary = dictionary
            self.label_encoder = label_encoder
            self.device = device
            self.max_length = max_length
        
        def predict(self, text):
            self.model.eval()
            with torch.no_grad():
                # Get LDA features
                bow = self.dictionary.doc2bow(text.split())
                topic_probs = [0] * config['num_lda_topics']
                for topic, prob in self.lda_model.get_document_topics(bow):
                    topic_probs[topic] = prob
                
                # Get BERT features
                tokens = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                lda_features = torch.tensor([topic_probs], dtype=torch.float).to(self.device)
                
                outputs = self.model(input_ids, attention_mask, lda_features)
                _, predicted = torch.max(outputs, 1)
                
                return {
                    'label': self.label_encoder.inverse_transform(predicted.cpu().numpy())[0],
                    'probabilities': torch.softmax(outputs, dim=1).cpu().numpy()[0]
                }
    
    # Create predictor instance
    predictor = ModelPredictor(model, tokenizer, lda_model, dictionary, label_encoder, device)
    
    return predictor, config

def delete_checkpoints(checkpoint_dir='.', filename_pattern='checkpoint_epoch_*.pt'):
    # If the directory is the current repo, use '.'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, filename_pattern))
    
    for file_path in checkpoint_files:
        try:
            os.remove(file_path)
            print(f"Deleted checkpoint: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

def main():
    # Configuration
    CLICKBAIT_PATH = "./train_data/train_clickbait.txt"
    NOT_CLICKBAIT_PATH = "./train_data/train_not_clickbait.txt"
    num_topics = 10
    batch_size = 4
    accumulation_steps = 4
    num_epochs = 10
    learning_rate = 1e-5
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_texts, test_texts, train_labels, test_labels, label_encoder = load_and_prepare_data(
        CLICKBAIT_PATH, NOT_CLICKBAIT_PATH)
    
    # Create LDA model
    dictionary = corpora.Dictionary([text.split() for text in train_texts])
    corpus = [dictionary.doc2bow(text.split()) for text in train_texts]
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, 
                            num_topics=num_topics, workers=2,
                            batch=True)
    
    del corpus
    clear_memory()
    
    # Initialize model components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = LDABertClassifier(bert_model, num_topics).to(device)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, dictionary, 
                               lda_model, num_topics)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, dictionary, 
                              lda_model, num_topics)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    model = train_model(train_loader, test_loader, model, device, num_epochs, 
                       learning_rate, accumulation_steps)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    print("\nFinal Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    if metrics['accuracy'] > 0.8 and metrics['f1'] > 0.8:
        print("Model training successful!")
    else:
        print("Model training needs improvement.")
    
    # Save model
    config = {
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
    
    save_model(model, tokenizer, lda_model, dictionary, label_encoder, config, metrics)
    print("\nModel saved successfully!")
    print(f"Model files saved in: {os.path.abspath('saved_model')}")
    
    delete_checkpoints(checkpoint_dir='./checkpoints', filename_pattern='checkpoint_epoch_*.pt')

if __name__ == "__main__":
    main()
