# Assessment 3: Training Pipeline and Data Preparation
**Time Limit: 60 minutes**
**Total Points: 100**

## Instructions
- Answer all questions in the format specified
- For True/False: Write "TRUE" or "FALSE"
- For Multiple Choice: Write the letter (A, B, C, D, E)
- For Free Response: Provide detailed explanations with code examples
- For Pipeline Design: Describe complete workflows with justifications

---

## Section A: True/False (20 points, 2 points each)

**Answer Format: TRUE or FALSE**

1. Configuration files should contain hardcoded file paths for better reproducibility.

2. Data validation should occur before any preprocessing steps in the pipeline.

3. Using the same random seed for train/validation splits ensures reproducible results.

4. Cross-validation is always better than a simple train/validation/test split.

5. The DataLoader's num_workers parameter should always be set to the number of CPU cores.

6. Gradient accumulation allows you to simulate larger batch sizes with limited memory.

7. Early stopping should always restore the best weights from the validation set.

8. Mixed precision training can reduce memory usage without significantly impacting accuracy.

9. Experiment tracking is only necessary for research, not production model development.

10. Time-based data splits are crucial when working with temporal data to avoid data leakage.

---

## Section B: Multiple Choice (25 points, 2.5 points each)

**Answer Format: Letter (A, B, C, D, or E)**

11. What is the correct order for a production training pipeline?
    A) Train model → Validate data → Preprocess → Split data
    B) Split data → Validate data → Preprocess → Train model
    C) Validate data → Split data → Preprocess → Train model
    D) Preprocess → Validate data → Split data → Train model
    E) Validate data → Preprocess → Split data → Train model

12. Which data validation check is most critical for text classification?
    A) Checking for duplicate texts
    B) Verifying all labels are present in training set
    C) Ensuring consistent text encoding
    D) Validating text length distribution
    E) All of the above are equally critical

13. For imbalanced datasets with a 90:10 class ratio, which approach is most effective?
    A) Oversample the minority class only
    B) Undersample the majority class only
    C) Use weighted loss functions
    D) Collect more data for minority class
    E) Use stratified sampling with weighted loss

14. What is the primary advantage of using configuration files (YAML/JSON) over hardcoded parameters?
    A) Faster execution speed
    B) Better memory efficiency
    C) Reproducibility and easy hyperparameter tracking
    D) Automatic parameter validation
    E) Reduced code complexity

15. When implementing gradient accumulation with accumulation_steps=4, how should you handle the loss?
    A) Multiply loss by 4
    B) Divide loss by 4
    C) Leave loss unchanged
    D) Square the loss
    E) Take absolute value of loss

16. Which collate function approach is best for variable-length text sequences?
    A) Pad all sequences to maximum length in dataset
    B) Truncate all sequences to minimum length
    C) Pad sequences to maximum length in current batch
    D) Use fixed length for all batches
    E) Don't use padding at all

17. For cross-validation in production ML pipelines, what's the main consideration?
    A) Always use 10-fold CV
    B) Ensure temporal consistency if data has time component
    C) Use as many folds as possible
    D) Cross-validation is not suitable for production
    E) Only use CV for hyperparameter tuning

18. What's the most important aspect of experiment tracking for production systems?
    A) Tracking every single hyperparameter
    B) Recording model weights at each epoch
    C) Tracking reproducible configuration and key metrics
    D) Saving all intermediate outputs
    E) Tracking only the final accuracy

19. When should you implement custom Dataset classes instead of using built-in ones?
    A) Always, for better performance
    B) When you need specific data loading or preprocessing logic
    C) Never, built-in classes are sufficient
    D) Only for very large datasets
    E) Only for text data

20. For memory-efficient training with large models, which technique is most effective?
    A) Reduce batch size only
    B) Use gradient checkpointing and mixed precision
    C) Increase learning rate
    D) Use more epochs with smaller datasets
    E) Disable all optimizations

---

## Section C: Pipeline Design (30 points, 6 points each)

**Answer Format: Detailed workflow descriptions with justifications**

21. **Production Data Pipeline Design (6 points)**
Design a complete data pipeline for a text classification system that processes customer support tickets. The system needs to handle:
- 100,000 new tickets per day
- Real-time classification requirements
- Model retraining weekly with new data

**Required components to specify:**
- Data ingestion and validation strategy
- Preprocessing and storage approach
- Training data preparation workflow
- Quality assurance checkpoints

22. **Experiment Management System (6 points)**
Design an experiment tracking system for a team of 5 ML engineers working on the same model. Requirements:
- Track model versions and performance
- Compare different architectures
- Reproduce any experiment
- Share results across team

**Required specifications:**
- Configuration management approach
- Metrics tracking strategy
- Model artifact storage
- Collaboration workflow

23. **Cross-Validation Strategy (6 points)**
Design a cross-validation strategy for a time-series sentiment analysis task where you need to:
- Predict customer sentiment for the next month
- Use 2 years of historical data
- Avoid temporal data leakage
- Validate model stability over time

**Required elements:**
- Splitting methodology
- Validation approach
- Temporal considerations
- Performance evaluation strategy

24. **Data Quality Assurance (6 points)**
Design a comprehensive data quality assurance system for training data that automatically:
- Detects data drift
- Identifies annotation errors
- Validates data format consistency
- Monitors label distribution changes

**Required components:**
- Automated validation checks
- Drift detection methodology
- Error reporting system
- Remediation workflows

25. **Scalable Training Infrastructure (6 points)**
Design a training infrastructure that can:
- Handle datasets from 1K to 10M examples
- Scale compute resources automatically
- Support multiple concurrent experiments
- Ensure reproducible results

**Required architecture:**
- Resource scaling strategy
- Job orchestration approach
- Storage and compute separation
- Cost optimization considerations

---

## Section D: Code Implementation (25 points, 5 points each)

**Answer Format: Complete, executable code**

26. **Configuration Management (5 points)**
Complete the configuration system that supports nested configurations and environment overrides:

```python
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

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data_path: str

def load_config(config_path: str, env_overrides: dict = None) -> Config:
    """Load config with optional environment overrides"""
    with open(config_path, 'r') as f:
        config_dict = ________________

    # Apply environment overrides
    if env_overrides:
        ________________

    return Config(
        model=________________,
        training=________________,
        data_path=________________
    )
```

27. **Data Validation System (5 points)**
Complete the data validation class that checks for common issues:

```python
class DataValidator:
    def __init__(self, required_columns: list, text_column: str, label_column: str):
        self.required_columns = required_columns
        self.text_column = text_column
        self.label_column = label_column

    def validate_dataset(self, df: pd.DataFrame) -> dict:
        """Comprehensive dataset validation"""
        issues = []

        # Check required columns
        missing_cols = ________________
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for null values
        null_counts = ________________
        if null_counts.any():
            issues.append(f"Null values: {null_counts.to_dict()}")

        # Check text length distribution
        text_lengths = ________________
        if text_lengths.max() > 1000:  # Flag very long texts
            issues.append("Very long texts detected")

        # Check label distribution
        label_dist = ________________
        imbalance_ratio = ________________
        if imbalance_ratio > 10:
            issues.append(f"Severe class imbalance: {imbalance_ratio:.1f}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': {
                'total_rows': len(df),
                'text_length_stats': ________________,
                'label_distribution': ________________
            }
        }
```

28. **Custom DataLoader with Collation (5 points)**
Complete the custom dataset and collate function for variable-length text:

```python
class TextDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return ________________

    def __getitem__(self, idx):
        text = ________________
        label = ________________

        # Tokenize
        encoding = ________________

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function for variable-length sequences"""
    input_ids = ________________
    attention_mask = ________________
    labels = ________________

    # Pad to max length in batch
    input_ids = ________________
    attention_mask = ________________

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
```

29. **Training Loop with Gradient Accumulation (5 points)**
Complete the training loop with gradient accumulation and mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, dataloader, optimizer, accumulation_steps=4):
    model.train()
    scaler = GradScaler()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        with autocast():
            outputs = ________________
            loss = ________________ / accumulation_steps

        # Backward pass with scaling
        ________________

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            ________________  # Unscale gradients
            ________________  # Clip gradients
            ________________  # Optimizer step
            ________________  # Update scaler
            ________________  # Zero gradients

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)
```

30. **Experiment Tracking Integration (5 points)**
Complete the experiment tracker that logs metrics and artifacts:

```python
class ExperimentTracker:
    def __init__(self, experiment_name: str, config: dict):
        self.experiment_name = experiment_name
        self.config = config
        self.metrics = []

    def log_config(self):
        """Log experiment configuration"""
        ________________

    def log_metrics(self, epoch: int, metrics: dict):
        """Log training metrics"""
        metrics_entry = {
            'epoch': epoch,
            'timestamp': ________________,
            **metrics
        }
        self.metrics.append(metrics_entry)

        # Also log to external service (e.g., wandb)
        ________________

    def save_model(self, model, path: str, is_best: bool = False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'model_state_dict': ________________,
            'config': self.config,
            'metrics': self.metrics,
            'experiment_name': self.experiment_name
        }

        torch.save(checkpoint, path)

        if is_best:
            ________________  # Save as best model

    def get_best_metric(self, metric_name: str, mode: str = 'max'):
        """Get best value of a metric"""
        if not self.metrics:
            return None

        values = [m.get(metric_name) for m in self.metrics if metric_name in m]
        if not values:
            return None

        return ________________ if mode == 'max' else ________________
```

---

## Scoring Rubric

### True/False (20 points)
- 2 points per correct answer

### Multiple Choice (25 points)
- 2.5 points per correct answer

### Pipeline Design (30 points)
- 6 points per question:
  - 6 points: Complete, detailed design with clear justifications
  - 5 points: Good design with minor gaps
  - 4 points: Adequate design showing understanding
  - 3 points: Basic design with some issues
  - 2 points: Minimal understanding shown
  - 0-1 points: Incorrect or missing

### Code Implementation (25 points)
- 5 points per question:
  - 5 points: Completely correct and production-ready
  - 4 points: Mostly correct with minor issues
  - 3 points: Shows understanding but has errors
  - 2 points: Partially correct
  - 1 point: Minimal understanding
  - 0 points: Incorrect or missing

### Grade Scale
- A: 90-100 points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points