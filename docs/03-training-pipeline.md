# Training Pipeline and Data Preparation

## Production-Ready Training Pipeline

### Project Structure
```
ml_project/
├── data/
│   ├── raw/              # Original data files
│   ├── processed/        # Cleaned and preprocessed data
│   └── splits/           # train/val/test splits
├── src/
│   ├── data/             # Data processing modules
│   ├── models/           # Model definitions
│   ├── training/         # Training scripts
│   └── evaluation/       # Evaluation utilities
├── configs/              # Training configurations
├── experiments/          # Experiment tracking
├── models/               # Saved model checkpoints
└── scripts/              # Training and inference scripts
```

### Configuration Management
```python
# config.yaml
model:
  name: "bert-base-uncased"
  num_labels: 3
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 5
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 1.0

data:
  train_file: "data/processed/train.jsonl"
  val_file: "data/processed/val.jsonl"
  test_file: "data/processed/test.jsonl"
  max_length: 512

# config.py
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    num_labels: int
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

@dataclass
class DataConfig:
    train_file: str
    val_file: str
    test_file: str
    max_length: int = 512

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data'])
    )
```

## Data Processing Pipeline

### Raw Data Validation
```python
import pandas as pd
from typing import List, Dict, Any
import json

class DataValidator:
    def __init__(self, required_columns: List[str]):
        self.required_columns = required_columns

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate data file and return statistics"""
        issues = []

        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)

        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts.to_dict()}")

        # Check data types
        text_cols = [col for col in df.columns if 'text' in col.lower()]
        for col in text_cols:
            if df[col].dtype != 'object':
                issues.append(f"Column {col} should be text but is {df[col].dtype}")

        # Statistics
        stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'issues': issues
        }

        if 'text' in df.columns:
            stats['text_lengths'] = {
                'mean': df['text'].str.len().mean(),
                'median': df['text'].str.len().median(),
                'max': df['text'].str.len().max(),
                'min': df['text'].str.len().min()
            }

        return stats

# Usage
validator = DataValidator(['text', 'label'])
stats = validator.validate_file('data/raw/dataset.csv')
print(f"Data validation results: {json.dumps(stats, indent=2)}")
```

### Data Preprocessing
```python
import re
from typing import List, Dict
import torch
from transformers import AutoTokenizer

class TextPreprocessor:
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (optional, depends on task)
        # text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def preprocess_batch(self, texts: List[str], labels: List[int] = None) -> Dict:
        """Preprocess a batch of texts"""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Tokenize
        encoding = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

        if labels is not None:
            result['labels'] = torch.tensor(labels, dtype=torch.long)

        return result

# Usage
preprocessor = TextPreprocessor('bert-base-uncased', max_length=512)
batch = preprocessor.preprocess_batch(
    texts=["This is great!", "Not good at all"],
    labels=[1, 0]
)
```

### Dataset Classes
```python
from torch.utils.data import Dataset, DataLoader
import json

class TextClassificationDataset(Dataset):
    def __init__(self, file_path: str, preprocessor: TextPreprocessor, label_map: Dict[str, int] = None):
        self.preprocessor = preprocessor
        self.data = self._load_data(file_path)
        self.label_map = label_map or self._create_label_map()

    def _load_data(self, file_path: str) -> List[Dict]:
        """Load data from file"""
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                return [json.loads(line) for line in f]
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _create_label_map(self) -> Dict[str, int]:
        """Create mapping from label strings to integers"""
        unique_labels = list(set(item['label'] for item in self.data))
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Preprocess text
        processed = self.preprocessor.preprocess_batch(
            texts=[item['text']],
            labels=[self.label_map[item['label']]]
        )

        # Remove batch dimension
        return {
            'input_ids': processed['input_ids'].squeeze(0),
            'attention_mask': processed['attention_mask'].squeeze(0),
            'labels': processed['labels'].squeeze(0)
        }

# Custom collate function for DataLoader
def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Usage
dataset = TextClassificationDataset('data/processed/train.jsonl', preprocessor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
```

## Training Loop Implementation

### Training Manager
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
import wandb
from tqdm import tqdm
import os
from typing import Dict, Any

class TrainingManager:
    def __init__(self, config: Config, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.name,
            num_labels=config.model.num_labels
        ).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        self.scheduler = None  # Will be set after calculating total steps

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        # Best model tracking
        self.best_val_score = 0.0
        self.best_model_path = None

    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )

        return {
            'val_loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = f"experiments/{self.experiment_name}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = f"{checkpoint_dir}/best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        # Setup scheduler
        total_steps = len(train_loader) * self.config.training.num_epochs
        self.setup_scheduler(total_steps)

        # Initialize wandb (optional)
        # wandb.init(project="text-classification", name=self.experiment_name)

        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Log metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            # wandb.log({
            #     'epoch': epoch,
            #     'train_loss': train_loss,
            #     **val_metrics
            # })

            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_val_score
            if is_best:
                self.best_val_score = val_metrics['f1']

            self.save_checkpoint(epoch, val_metrics, is_best)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['val_loss'])
            self.val_metrics.append(val_metrics)

        print(f"\nTraining completed! Best F1 score: {self.best_val_score:.4f}")
        print(f"Best model saved at: {self.best_model_path}")

        return self.best_model_path
```

## Data Splitting Strategies

### Time-Based Splits
```python
def time_based_split(df: pd.DataFrame, date_column: str, train_ratio: float = 0.8):
    """Split data based on time to avoid data leakage"""
    df_sorted = df.sort_values(date_column)
    split_index = int(len(df_sorted) * train_ratio)

    train_df = df_sorted[:split_index]
    val_df = df_sorted[split_index:]

    return train_df, val_df
```

### Stratified Splits
```python
from sklearn.model_selection import train_test_split

def stratified_split(df: pd.DataFrame, label_column: str, test_size: float = 0.2):
    """Ensure balanced representation of all classes"""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=42
    )
    return train_df, test_df
```

### Cross-Validation Setup
```python
from sklearn.model_selection import StratifiedKFold

def setup_cross_validation(df: pd.DataFrame, label_column: str, n_folds: int = 5):
    """Setup for k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds = []
    for train_idx, val_idx in skf.split(df, df[label_column]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        folds.append((train_df, val_df))

    return folds
```

## Experiment Tracking

### Simple Experiment Logger
```python
import json
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        self.log_file = os.path.join(experiment_dir, 'experiment_log.json')
        self.metrics_file = os.path.join(experiment_dir, 'metrics.json')

    def log_config(self, config: Config):
        """Log experiment configuration"""
        config_dict = {
            'timestamp': datetime.now().isoformat(),
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'data': config.data.__dict__
        }

        with open(self.log_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict[str, float]):
        """Log training metrics"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = []

        metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
```

## Common Pipeline Issues and Solutions

### 1. Memory Issues
```python
# Solution: Gradient accumulation
def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps  # Scale loss
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### 2. Data Loading Bottlenecks
```python
# Solution: Optimized DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Use multiple processes
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2  # Prefetch batches
)
```

### 3. Inconsistent Results
```python
# Solution: Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 4. Training Script Example
```python
# train.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Load config
    config = load_config(args.config)

    # Setup data
    preprocessor = TextPreprocessor(config.model.name, config.data.max_length)

    train_dataset = TextClassificationDataset(config.data.train_file, preprocessor)
    val_dataset = TextClassificationDataset(config.data.val_file, preprocessor, train_dataset.label_map)

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Setup training
    trainer = TrainingManager(config, args.experiment_name)

    # Train
    best_model_path = trainer.train(train_loader, val_loader)

    print(f"Training completed. Best model: {best_model_path}")

if __name__ == "__main__":
    main()
```

## Next Steps

Continue to [Evaluation and Optimization](04-evaluation-optimization.md) to learn how to properly evaluate and improve your trained models.