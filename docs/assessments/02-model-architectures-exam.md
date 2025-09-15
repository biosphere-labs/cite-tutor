# Assessment 2: Model Architectures and Selection
**Time Limit: 50 minutes**
**Total Points: 100**

## Instructions
- Answer all questions in the format specified
- For True/False: Write "TRUE" or "FALSE"
- For Multiple Choice: Write the letter (A, B, C, D, E)
- For Free Response: Provide detailed explanations with code examples where requested
- For Architecture Design: Draw/describe architectures and justify choices

---

## Section A: True/False (20 points, 2 points each)

**Answer Format: TRUE or FALSE**

1. Encoder-only models like BERT are best suited for text generation tasks.

2. The attention mechanism allows models to focus on relevant parts of the input sequence.

3. Larger models always perform better than smaller models on any given task.

4. LoRA (Low-Rank Adaptation) reduces the number of trainable parameters during fine-tuning.

5. Vision Transformers (ViTs) generally outperform CNNs on small image datasets.

6. In transformer architecture, the number of attention heads must equal the hidden dimension size.

7. Decoder-only models like GPT can only generate text and cannot be used for classification.

8. Model quantization always reduces model accuracy significantly.

9. Gradient checkpointing trades computation time for memory usage.

10. The pooler output from BERT represents the entire sequence in a single vector.

---

## Section B: Multiple Choice (30 points, 3 points each)

**Answer Format: Letter (A, B, C, D, or E)**

11. Which architecture is most suitable for sentiment classification of customer reviews?
    A) GPT (decoder-only)
    B) T5 (encoder-decoder)
    C) BERT (encoder-only)
    D) CNN with LSTM
    E) Traditional RNN

12. What is the main advantage of attention mechanisms over traditional RNNs?
    A) Faster training on CPUs
    B) Ability to process sequences in parallel
    C) Lower memory requirements
    D) Better performance on short sequences
    E) Simpler architecture

13. For a task requiring both understanding input and generating output (like summarization), which architecture is most appropriate?
    A) Encoder-only (BERT-style)
    B) Decoder-only (GPT-style)
    C) Encoder-decoder (T5-style)
    D) CNN-based
    E) RNN-based

14. What happens when you use a model size that's too large for your dataset?
    A) Training becomes faster
    B) Model will definitely overfit
    C) Memory usage decreases
    D) Inference becomes slower but accuracy always improves
    E) The model cannot be trained

15. In LoRA fine-tuning, what do the rank parameters (r) control?
    A) The learning rate
    B) The number of attention heads
    C) The dimensionality of the low-rank matrices
    D) The sequence length
    E) The batch size

16. Which statement about model compression is most accurate?
    A) Pruning always improves inference speed
    B) Quantization reduces model size but never affects accuracy
    C) Distillation requires a larger teacher model
    D) All compression techniques work equally well for all architectures
    E) Compression is only useful for deployment, not training

17. When choosing between BERT-base and BERT-large for a production system, what should be your primary consideration?
    A) Always choose the larger model for better accuracy
    B) Balance accuracy requirements with computational constraints
    C) Choose based on the size of your training data only
    D) BERT-large is always faster
    E) The choice doesn't matter for production systems

18. What is the primary purpose of the [CLS] token in BERT?
    A) To separate sentences
    B) To mark the end of sequences
    C) To provide a sequence-level representation
    D) To handle unknown words
    E) To control attention patterns

19. In transformer architecture, what does "multi-head attention" accomplish?
    A) Processes multiple sequences simultaneously
    B) Allows the model to attend to different representation subspaces
    C) Reduces computational complexity
    D) Eliminates the need for positional encoding
    E) Increases the vocabulary size

20. For named entity recognition (NER), which model output should you use?
    A) Pooler output
    B) [CLS] token representation
    C) Token-level hidden states
    D) Attention weights
    E) Final layer embeddings only

---

## Section C: Architecture Design (25 points, 5 points each)

**Answer Format: Describe architecture components and justify choices**

21. **Text Classification Architecture (5 points)**
Design a model architecture for classifying product reviews into 5 sentiment categories (very negative, negative, neutral, positive, very positive). You have 100,000 training examples.

**Required components to specify:**
- Base model choice (with justification)
- Classification head design
- Input preprocessing approach
- Expected model size considerations

22. **Multi-task Architecture (5 points)**
Design an architecture that can simultaneously perform:
- Sentiment analysis (3 classes)
- Intent detection (10 classes)
- Named entity recognition (7 entity types)

**Required in your design:**
- Shared components
- Task-specific heads
- How you would handle different output formats
- Training strategy considerations

23. **Memory-Constrained Architecture (5 points)**
You need to deploy a text classification model on edge devices with only 2GB RAM. Design an architecture that maintains reasonable accuracy while meeting memory constraints.

**Required specifications:**
- Model size justification
- Specific optimization techniques
- Trade-offs you're making
- Deployment considerations

24. **Custom Domain Architecture (5 points)**
Design an architecture for a legal document classification system that needs to handle documents up to 10,000 words. Standard BERT has a 512 token limit.

**Required solutions:**
- How to handle long documents
- Architecture modifications needed
- Alternative approaches considered
- Performance implications

25. **Multilingual Architecture (5 points)**
Design an architecture for sentiment analysis that works across English, Spanish, and French, where you have labeled data only in English.

**Required components:**
- Base model selection rationale
- Cross-lingual transfer approach
- Training strategy
- Evaluation considerations

---

## Section D: Code Implementation (25 points, 5 points each)

**Answer Format: Complete code with proper syntax and imports**

26. **Custom Classification Head (5 points)**
Complete the custom classifier that adds dropout and multiple dense layers:

```python
class CustomClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, dropout_rate=0.1):
        super().__init__()
        self.backbone = ________________
        self.dropout = ________________
        self.intermediate = ________________  # 768 -> 256
        self.classifier = ________________   # 256 -> num_classes

    def forward(self, input_ids, attention_mask=None):
        outputs = ________________
        pooled_output = ________________
        x = self.dropout(pooled_output)
        x = ________________  # Apply intermediate layer
        x = self.dropout(x)
        logits = ________________
        return logits
```

27. **LoRA Implementation Setup (5 points)**
Complete the LoRA configuration and model setup:

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA parameters
lora_config = LoraConfig(
    r=________________,  # Rank
    lora_alpha=________________,
    target_modules=________________,  # Which layers to apply LoRA
    lora_dropout=________________,
)

# Apply LoRA to model
model = ________________
model = ________________

# Count trainable parameters
trainable_params = ________________
total_params = ________________
print(f"Trainable: {trainable_params}, Total: {total_params}")
```

28. **Multi-Task Model (5 points)**
Complete the multi-task model implementation:

```python
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, sentiment_classes, intent_classes, ner_classes):
        super().__init__()
        self.backbone = ________________

        # Task-specific heads
        self.sentiment_head = ________________
        self.intent_head = ________________
        self.ner_head = ________________

    def forward(self, input_ids, attention_mask, task_type):
        outputs = ________________

        if task_type == 'sentiment':
            return ________________
        elif task_type == 'intent':
            return ________________
        elif task_type == 'ner':
            return ________________
```

29. **Model Compression (5 points)**
Complete the dynamic quantization implementation:

```python
import torch.quantization as quant

def quantize_model(model):
    # Set model to evaluation mode
    ________________

    # Apply dynamic quantization
    quantized_model = ________________

    return quantized_model

def compare_model_sizes(original_model, quantized_model):
    # Calculate model sizes
    original_size = ________________
    quantized_size = ________________

    compression_ratio = ________________
    return original_size, quantized_size, compression_ratio
```

30. **Architecture Selection Logic (5 points)**
Complete the architecture selection function:

```python
def select_architecture(task_type, data_size, latency_requirement, accuracy_requirement):
    """
    Select appropriate model architecture based on requirements
    """

    if task_type == 'classification':
        if data_size < 1000:
            return ________________
        elif latency_requirement == 'low' and accuracy_requirement == 'high':
            return ________________
        elif latency_requirement == 'high':  # Need fast inference
            return ________________
        else:
            return ________________

    elif task_type == 'generation':
        if latency_requirement == 'high':
            return ________________
        else:
            return ________________

    elif task_type == 'sequence_labeling':
        return ________________
```

---

## Scoring Rubric

### True/False (20 points)
- 2 points per correct answer
- 0 points for incorrect answers

### Multiple Choice (30 points)
- 3 points per correct answer
- 0 points for incorrect answers

### Architecture Design (25 points)
- 5 points per question:
  - 5 points: Complete design with clear justification and technical understanding
  - 4 points: Good design with minor gaps in justification
  - 3 points: Basic design showing understanding but missing key components
  - 2 points: Partial design with some technical understanding
  - 1 point: Minimal understanding shown
  - 0 points: Incorrect or missing

### Code Implementation (25 points)
- 5 points per question:
  - 5 points: Completely correct code that would execute properly
  - 4 points: Mostly correct with minor syntax issues
  - 3 points: Shows understanding but has implementation errors
  - 2 points: Partially correct with significant issues
  - 1 point: Shows minimal understanding
  - 0 points: Incorrect or missing

### Grade Scale
- A: 90-100 points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points

### Answer Key Notes
*Provided separately for instructor use*