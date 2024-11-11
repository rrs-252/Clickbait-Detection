import os
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim import corpora
from gensim.models import LdaMulticore
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import glob

# Import HTMLParserPreprocessor from the separate file
from html_parser_preprocessor import HTMLParserPreprocessor

# Memory management functions
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
def print_gpu_memory():
    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB')
    print(f'GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB')

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
        if chunk:
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
        
        bow = self.dictionary.doc2bow(text.split())
        topic_probs = [0] * self.num_topics
        for topic, prob in self.lda_model.get_document_topics(bow):
            topic_probs[topic] = prob
            
        roberta_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': roberta_tokens['input_ids'].squeeze(),
            'attention_mask': roberta_tokens['attention_mask'].squeeze(),
            'lda_features': torch.tensor(topic_probs, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

class LDARoBERTaClassifier(nn.Module):
    def __init__(self, roberta_model, num_lda_features):
        super().__init__()
        self.roberta_model = roberta_model
        self.roberta_model.gradient_checkpointing_enable()
        self.num_lda_features = num_lda_features
        self.classifier = nn.Linear(roberta_model.config.hidden_size + num_lda_features, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, lda_features):
        roberta_output = self.roberta_model(input_ids, attention_mask=attention_mask)[0][:, 0, :]
        roberta_output = self.dropout(roberta_output)
        combined_features = torch.cat((roberta_output, lda_features), dim=1)
        return self.classifier(combined_features)

def prepare_data(clickbait_path, not_clickbait_path):
    texts = []
    labels = []

    # Create an instance of the HTMLParserPreprocessor
    html_parser = HTMLParserPreprocessor()

    # Preprocess clickbait data
    chunk_texts, chunk_labels = load_data_in_chunks(clickbait_path, "clickbait")
    texts.extend([html_parser.parse_and_extract(text) for text in chunk_texts])
    labels.extend(chunk_labels)
    clear_memory()

    # Preprocess non-clickbait data
    chunk_texts, chunk_labels = load_data_in_chunks(not_clickbait_path, "not clickbait")
    texts.extend([html_parser.parse_and_extract(text) for text in chunk_texts])
    labels.extend(chunk_labels)
    clear_memory()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42)

    return train_texts, test_texts, train_labels, test_labels, label_encoder

def train_model(train_loader, test_loader, model, device, num_epochs=10, learning_rate=1e-5,
                batch_size=4, accumulation_steps=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                lda_features = batch['lda_features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, lda_features)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    clear_memory()
                
                train_loss += loss.item() * accumulation_steps
                
                if i % 100 == 0:
                    print_gpu_memory()
                    
            except RuntimeError as e:
                print(f"Error in batch {i}: {e}")
                clear_memory()
                continue
        
        scheduler.step()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'lda_roberta_checkpoint_epoch_{epoch}.pt')
        
        clear_memory()
    
    return model, optimizer, scheduler

def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    y_true = []
    y_pred = []
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lda_features = batch['lda_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, lda_features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
            
            clear_memory()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'test_loss': test_loss / len(test_loader)
    }

    return metrics

def save_model(model, optimizer, scheduler, config, metrics, label_encoder, dictionary, lda_model, save_dir='saved_model_roberta'):
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = f'{save_dir}/lda_roberta_model'
    
    # Save model state and architecture
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': config
    }, f'{model_save_path}.pt')

    # Save LDA model and dictionary
    lda_model.save(f'{model_save_path}_lda.model')
    dictionary.save(f'{model_save_path}_dictionary.dict')

    # Save label encoder
    with open(f'{model_save_path}_label_encoder.npy', 'wb') as f:
        np.save(f, label_encoder.classes_)

    # Save configuration and metadata
    with open(f'{save_dir}/model_config.json', 'w') as f:
        json.dump({**config, 'model_performance': metrics}, f, indent=4)

    print(f"\nModel saved successfully in: {os.path.abspath(save_dir)}")
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preparation
    train_texts, test_texts, train_labels, test_labels, label_encoder = prepare_data(
        CLICKBAIT_PATH, NOT_CLICKBAIT_PATH)
    
    # LDA preparation
    dictionary = corpora.Dictionary([text.split() for text in train_texts])
    corpus = [dictionary.doc2bow(text.split()) for text in train_texts]
    num_topics = 10
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, 
                            num_topics=num_topics, workers=2,
                            batch=True)
    
    del corpus
    clear_memory()
    
    # Model initialization
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    model = LDARoBERTaClassifier(roberta_model, num_topics).to(device)
    
    # Dataset creation
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, dictionary, lda_model, num_topics)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, dictionary, lda_model, num_topics)
    
    # DataLoader creation
    batch_size = 4
    accumulation_steps = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training
    model, optimizer, scheduler = train_model(
        train_loader, test_loader, model, device,
        num_epochs=10, learning_rate=1e-5,
        batch_size=batch_size, accumulation_steps=accumulation_steps
    )
    
    # Evaluation
    metrics = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss())
    
    print("\nFinal Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Save model
    config = {
        'training_params': {
            'batch_size': batch_size,
            'initial_learning_rate': 1e-5,
            'num_epochs': 10,
            'accumulation_steps': accumulation_steps,
            'max_grad_norm': 1.0,
        },
        'tokenizer_name': 'roberta-base',
        'max_length': train_dataset.max_length,
        'num_lda_topics': num_topics,
        'model_architecture': {
            'base_model': 'roberta-base',
            'dropout_rate': 0.1,
            'hidden_size': roberta_model.config.hidden_size,
        }
    }
    
    save_model(model, optimizer, scheduler, config, metrics, label_encoder, 
               dictionary, lda_model)
    
    delete_checkpoints(checkpoint_dir='.', filename_pattern='lda_roberta_checkpoint_epoch_*.pt')

if __name__ == "__main__":
    main()

# Function to load the saved model (for future use)
def load_saved_model(model_dir='./saved_model_roberta', device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Define file paths
    config_path = f'{model_dir}/model_config.json'
    model_path = f'{model_dir}/lda_roberta_model.pt'
    lda_path = f'{model_dir}/lda_roberta_model_lda.model'
    dict_path = f'{model_dir}/lda_roberta_model_dictionary.dict'
    label_encoder_path = f'{model_dir}/lda_roberta_model_label_encoder.npy'
    
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
    tokenizer = RobertaTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Load RoBERTa and initialize combined model
    print("Initializing RoBERTa model...")
    roberta_model = RobertaModel.from_pretrained(config['model_architecture']['base_model'])
    model = LDARoBERTaClassifier(roberta_model, config['num_lda_topics']).to(device)
    
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
                
                # Get RoBERTa features
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
