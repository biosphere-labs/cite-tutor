# Training Fundamentals for Developers

## From RAG to Training: The Mental Model Shift

### RAG vs Training
- **RAG (Retrieval-Augmented Generation)**: Using pre-trained models with your data as context
- **Training**: Teaching a model to learn patterns from your data by adjusting its parameters

Think of RAG as giving a consultant your documents to reference, while training is like hiring an intern and teaching them your domain expertise.

## Core Training Concepts

### What Actually Happens During Training

```python
# Simplified training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass: model makes predictions
        predictions = model(batch.inputs)

        # Calculate how wrong the predictions are
        loss = loss_function(predictions, batch.targets)

        # Backpropagation: adjust model weights to reduce error
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Key Components

#### 1. Dataset
- **Training Set**: Data used to teach the model
- **Validation Set**: Data used to tune hyperparameters
- **Test Set**: Data used for final evaluation (never seen during training)

#### 2. Loss Function
Measures how wrong your model's predictions are:
- **Classification**: CrossEntropyLoss, BCELoss
- **Regression**: MSELoss, L1Loss
- **Custom**: Define your own based on business logic

#### 3. Optimizer
Updates model weights based on gradients:
- **Adam**: Good default choice, adaptive learning rates
- **SGD**: Simple, requires manual learning rate tuning
- **AdamW**: Adam with weight decay, popular for transformers

#### 4. Learning Rate
How big steps the model takes when learning:
- Too high: Model oscillates or diverges
- Too low: Training is slow or gets stuck
- **Learning Rate Scheduling**: Start high, decrease over time

## Types of Training

### 1. Training from Scratch
```python
# Initialize random weights
model = MyModel()
# Train on your data
trainer = Trainer(model, train_data, val_data)
trainer.train()
```
- **Pros**: Model learns exactly your patterns
- **Cons**: Requires lots of data and compute
- **Use when**: You have massive datasets and unique requirements

### 2. Fine-tuning (Most Common)
```python
# Load pre-trained model
model = transformers.AutoModel.from_pretrained('bert-base-uncased')
# Replace final layer for your task
model.classifier = nn.Linear(768, num_classes)
# Train with lower learning rate
trainer = Trainer(model, train_data, learning_rate=1e-5)
```
- **Pros**: Faster, requires less data
- **Cons**: Inherits biases from pre-trained model
- **Use when**: Your task is similar to what model was originally trained on

### 3. Transfer Learning
```python
# Freeze early layers
for param in model.encoder.parameters():
    param.requires_grad = False
# Only train final layers
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
```
- **Pros**: Very fast training
- **Cons**: Limited adaptation
- **Use when**: You have little data or compute

## Data Preparation

### Data Quality > Data Quantity
```python
# Bad: Inconsistent, noisy data
train_data = [
    {"text": "gr8 product!!!", "label": "positive"},
    {"text": "Terrible quality.", "label": "negative"},
    {"text": "ok i guess", "label": "neutral"},  # Ambiguous
]

# Good: Clean, consistent data
train_data = [
    {"text": "Great product! Highly recommend.", "label": "positive"},
    {"text": "Poor quality, not worth the price.", "label": "negative"},
    {"text": "Average product, meets expectations.", "label": "neutral"},
]
```

### Data Splits
```python
from sklearn.model_selection import train_test_split

# 80/10/10 split is common
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
```

### Handling Imbalanced Data
```python
# Check class distribution
print(train_data['label'].value_counts())

# Solutions:
# 1. Weighted loss
class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

# 2. Oversampling minority class
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 3. Undersample majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

## Common Training Patterns

### 1. Supervised Learning
You have input-output pairs:
```python
# Text classification
{"text": "This movie is amazing", "label": "positive"}

# Named Entity Recognition
{"text": "John works at Google", "entities": [{"start": 0, "end": 4, "label": "PERSON"}]}
```

### 2. Self-Supervised Learning
Model learns from the data structure itself:
```python
# Masked Language Modeling (BERT-style)
original = "The cat sat on the mat"
masked = "The [MASK] sat on the mat"
# Model learns to predict "cat"

# Next Token Prediction (GPT-style)
input_sequence = "The cat sat on the"
target = "mat"
# Model learns to predict next token
```

### 3. Few-Shot Learning
Learn with minimal examples:
```python
# Prompt engineering with examples
prompt = """
Examples:
Text: "Great service!" -> Sentiment: Positive
Text: "Terrible experience" -> Sentiment: Negative
Text: "It was okay" -> Sentiment: Neutral

Text: "Amazing product!" -> Sentiment:
"""
```

## Training Monitoring

### Essential Metrics to Track
```python
# During training, log:
metrics = {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy,
    'learning_rate': current_lr,
    'epoch': epoch,
    'step': global_step
}

# Use tools like wandb, tensorboard, or simple logging
import wandb
wandb.log(metrics)
```

### Signs of Good Training
- Training loss decreases steadily
- Validation loss follows training loss (small gap)
- Metrics improve on validation set
- No signs of overfitting

### Red Flags
- Validation loss increases while training loss decreases (overfitting)
- Loss doesn't decrease at all (learning rate too high/low)
- Loss explodes (gradient explosion)
- Training is unstable (oscillating losses)

## Practical Tips

### Start Simple
```python
# Begin with a simple baseline
baseline_model = LogisticRegression()
baseline_model.fit(X_train, y_train)
baseline_score = baseline_model.score(X_val, y_val)

# Your neural network should beat this
```

### Debugging Strategy
1. **Overfit on small dataset**: Ensure model can learn
2. **Check data loading**: Print batches, verify labels
3. **Verify loss calculation**: Manual calculation vs. model output
4. **Gradient checking**: Ensure gradients flow properly

### Hyperparameter Tuning
```python
# Start with learning rate
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

# Then batch size
batch_sizes = [16, 32, 64, 128]

# Finally architecture choices
hidden_sizes = [128, 256, 512]
```

## Common Pitfalls for Senior Developers

### 1. Over-engineering Early
- Start with simple models and baseline approaches
- Don't optimize prematurely
- Prove the concept works before scaling

### 2. Ignoring Data Quality
- Your model is only as good as your data
- Spend time on data cleaning and validation
- Bad data will make the best model fail

### 3. Not Establishing Baselines
- Always compare against simple heuristics
- Random baseline, majority class, simple ML models
- Your deep learning model should significantly outperform these

### 4. Training Without Validation
- Always hold out validation data
- Monitor overfitting continuously
- Use early stopping to prevent overfitting

## Next Steps

Move on to [Model Architectures](02-model-architectures.md) to understand how to choose and configure the right model for your task.