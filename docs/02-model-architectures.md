# Model Architectures and Selection Guide

## Architecture Decision Framework

### Choose Based on Your Data Type

#### Text Data
- **Transformers** (BERT, GPT, T5): Current gold standard
- **RNNs/LSTMs**: Legacy, still useful for sequential data with memory constraints
- **CNNs**: Fast for text classification, less common now

#### Image Data
- **Convolutional Neural Networks (CNNs)**: ResNet, EfficientNet, Vision Transformers
- **Vision Transformers (ViTs)**: Increasingly popular, especially with large datasets

#### Tabular Data
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost (often best choice)
- **Neural Networks**: TabNet, deep feedforward networks
- **Traditional ML**: Random Forest, SVM (good baselines)

#### Time Series
- **Transformers**: For long sequences
- **LSTMs/GRUs**: For shorter sequences
- **CNN-LSTM**: Hybrid approaches

## Understanding Transformers (Most Important for Modern AI)

### The Attention Mechanism
```python
# Simplified attention calculation
def attention(query, key, value):
    # How much should we pay attention to each position?
    scores = torch.matmul(query, key.transpose(-2, -1))
    weights = torch.softmax(scores, dim=-1)
    # Weighted combination of values
    output = torch.matmul(weights, value)
    return output
```

### Encoder vs Decoder Architectures

#### Encoder-Only (BERT-style)
- **Use for**: Classification, named entity recognition, sentiment analysis
- **Characteristics**: Bidirectional context, good for understanding
- **Training**: Masked language modeling

```python
from transformers import AutoModel
# Load encoder-only model
model = AutoModel.from_pretrained('bert-base-uncased')
# Add classification head
classifier = nn.Linear(model.config.hidden_size, num_classes)
```

#### Decoder-Only (GPT-style)
- **Use for**: Text generation, completion, conversation
- **Characteristics**: Autoregressive, generates one token at a time
- **Training**: Next token prediction

```python
from transformers import AutoModelForCausalLM
# Load decoder-only model
model = AutoModelForCausalLM.from_pretrained('gpt2')
# Already has language modeling head
```

#### Encoder-Decoder (T5-style)
- **Use for**: Translation, summarization, question answering
- **Characteristics**: Input through encoder, output through decoder
- **Training**: Sequence-to-sequence tasks

```python
from transformers import AutoModelForSeq2SeqLM
# Load encoder-decoder model
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
```

## Model Size Considerations

### Parameters vs Performance Trade-offs

#### Small Models (< 100M parameters)
- **Examples**: DistilBERT, TinyBERT, GPT2-small
- **Pros**: Fast inference, low memory, easy to deploy
- **Cons**: Lower accuracy on complex tasks
- **Use when**: Resource constraints, simple tasks, real-time inference

#### Medium Models (100M - 1B parameters)
- **Examples**: BERT-base, GPT2-medium, RoBERTa-base
- **Pros**: Good balance of performance and efficiency
- **Cons**: Still requires significant compute
- **Use when**: Most production applications

#### Large Models (1B+ parameters)
- **Examples**: GPT-3, BERT-large, T5-large
- **Pros**: Best performance on complex tasks
- **Cons**: Expensive to train and run
- **Use when**: Performance is critical, have sufficient resources

### Choosing Model Size
```python
# Development workflow
# 1. Start with smallest model that makes sense
model_small = AutoModel.from_pretrained('distilbert-base-uncased')

# 2. Establish baseline performance
baseline_score = evaluate_model(model_small, val_data)

# 3. Try larger model if needed
model_medium = AutoModel.from_pretrained('bert-base-uncased')
improved_score = evaluate_model(model_medium, val_data)

# 4. Only go larger if improvement justifies cost
if improved_score - baseline_score > threshold:
    model_large = AutoModel.from_pretrained('bert-large-uncased')
```

## Fine-tuning Strategies

### Full Fine-tuning
```python
# All parameters are trainable
model = AutoModel.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=2e-5)
```
- **Pros**: Maximum adaptation to your task
- **Cons**: Most expensive, risk of overfitting
- **Use when**: Lots of data, very different domain

### Partial Fine-tuning (Layer Freezing)
```python
# Freeze early layers, train later ones
model = AutoModel.from_pretrained('bert-base-uncased')
for param in model.embeddings.parameters():
    param.requires_grad = False
for param in model.encoder.layer[:6].parameters():  # Freeze first 6 layers
    param.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
```
- **Pros**: Faster training, less overfitting
- **Cons**: Less adaptation capability
- **Use when**: Limited data, similar domain

### LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model

# Add small trainable matrices instead of training all weights
lora_config = LoraConfig(
    r=16,  # Rank of adaptation
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
```
- **Pros**: Very few trainable parameters, efficient
- **Cons**: Limited adaptation
- **Use when**: Large models, limited compute

## Architecture Selection Examples

### Text Classification
```python
# Task: Classify customer reviews
# Data: 10K labeled reviews

# Option 1: Start simple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Option 2: Modern approach
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3  # positive, negative, neutral
)

# Option 3: If you need highest accuracy
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-large',
    num_labels=3
)
```

### Named Entity Recognition
```python
# Task: Extract person names, organizations, locations
# Data: Annotated documents

# Good choice: BERT-based NER model
model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-cased',  # Cased is important for NER
    num_labels=len(label_list)
)

# For better performance
model = AutoModelForTokenClassification.from_pretrained(
    'roberta-base',
    num_labels=len(label_list)
)
```

### Text Generation
```python
# Task: Generate product descriptions
# Data: Product attributes -> descriptions

# Option 1: Fine-tune GPT-2
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Option 2: Use T5 for controlled generation
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

# Option 3: For best quality (if you have resources)
model = AutoModelForCausalLM.from_pretrained('gpt2-large')
```

## Custom Architecture Patterns

### Adding Task-Specific Heads
```python
class CustomClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, dropout_rate=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # or outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

### Multi-Task Learning
```python
class MultiTaskModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)

        # Multiple heads for different tasks
        self.sentiment_head = nn.Linear(768, 3)  # positive, negative, neutral
        self.intent_head = nn.Linear(768, 10)    # 10 different intents
        self.ner_head = nn.Linear(768, 7)        # BIO + entity types

    def forward(self, input_ids, attention_mask, task_type):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if task_type == 'sentiment':
            return self.sentiment_head(outputs.pooler_output)
        elif task_type == 'intent':
            return self.intent_head(outputs.pooler_output)
        elif task_type == 'ner':
            return self.ner_head(outputs.last_hidden_state)
```

## Performance Optimization

### Model Compression Techniques

#### Distillation
```python
# Train smaller model to mimic larger one
teacher_model = AutoModel.from_pretrained('bert-large-uncased')
student_model = AutoModel.from_pretrained('distilbert-base-uncased')

# Training loop with distillation loss
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.7):
    distill_loss = nn.KLDivLoss()(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)

    student_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    return alpha * distill_loss + (1 - alpha) * student_loss
```

#### Quantization
```python
# Reduce model precision for faster inference
import torch.quantization as quant

# Post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Quantization-aware training
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model)
# Train model_prepared, then convert
model_quantized = quant.convert(model_prepared)
```

### Memory Optimization
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Common Architecture Mistakes

### 1. Starting Too Complex
```python
# Bad: Jumping to complex architecture immediately
class OverEngineeredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('bert-large')
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(768, 12) for _ in range(6)
        ])
        self.complex_head = ComplexClassificationHead(768, 1000, 100)

# Good: Start simple, add complexity if needed
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_classes
)
```

### 2. Ignoring Task-Model Alignment
```python
# Bad: Using decoder-only model for classification
model = AutoModelForCausalLM.from_pretrained('gpt2')  # Wrong for classification

# Good: Use appropriate architecture
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
```

### 3. Not Considering Inference Requirements
```python
# Consider deployment constraints early
if inference_time_critical:
    model = AutoModel.from_pretrained('distilbert-base-uncased')
elif accuracy_critical:
    model = AutoModel.from_pretrained('roberta-large')
else:
    model = AutoModel.from_pretrained('bert-base-uncased')  # Good balance
```

## Next Steps

Continue to [Training Pipeline](03-training-pipeline.md) to learn how to set up efficient training workflows.