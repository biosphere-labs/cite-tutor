# Assessment 1: Training Fundamentals
**Time Limit: 45 minutes**
**Total Points: 100**

## Instructions
- Answer all questions in the format specified
- For True/False: Write "TRUE" or "FALSE"
- For Multiple Choice: Write the letter (A, B, C, D, E)
- For Free Response: Provide detailed explanations with code examples where requested
- For Code Completion: Complete the missing code segments

---

## Section A: True/False (20 points, 2 points each)

**Answer Format: TRUE or FALSE**

1. In RAG systems, you modify the model's internal parameters to learn from your data.

2. Fine-tuning requires less data than training from scratch because you start with pre-trained weights.

3. The learning rate determines how much the model weights change after each batch.

4. A high validation loss with low training loss typically indicates overfitting.

5. Adam optimizer automatically adjusts learning rates for each parameter.

6. Gradient accumulation allows you to effectively use larger batch sizes with limited memory.

7. Transfer learning involves freezing all layers except the final classification layer.

8. Cross-entropy loss is only suitable for binary classification problems.

9. The validation set should be used to make decisions about hyperparameters during training.

10. Mixed precision training always improves model accuracy compared to full precision.

---

## Section B: Multiple Choice (30 points, 3 points each)

**Answer Format: Letter (A, B, C, D, or E)**

11. What is the primary difference between RAG and custom model training?
    A) RAG uses larger models
    B) RAG retrieves information at inference time, training modifies model parameters
    C) RAG is faster during inference
    D) RAG requires more GPU memory
    E) RAG can only work with text data

12. When should you choose fine-tuning over training from scratch?
    A) When you have unlimited computational resources
    B) When your task is very different from the pre-trained model's original task
    C) When you have limited data and your task is similar to the pre-training task
    D) When you want the fastest possible inference
    E) Never, training from scratch is always better

13. What happens when the learning rate is too high?
    A) Training becomes very slow
    B) The model may overffit quickly
    C) Loss oscillates or diverges
    D) Gradients vanish
    E) Memory usage increases

14. Which loss function is most appropriate for multi-class classification?
    A) Mean Squared Error (MSE)
    B) Binary Cross-Entropy
    C) Categorical Cross-Entropy
    D) Huber Loss
    E) L1 Loss

15. What is the purpose of a validation set?
    A) To train the model
    B) To tune hyperparameters and monitor overfitting
    C) To test final model performance
    D) To augment training data
    E) To calculate gradients

16. In the training loop, what order should these operations occur?
    A) backward(), step(), zero_grad(), forward()
    B) forward(), backward(), step(), zero_grad()
    C) zero_grad(), forward(), backward(), step()
    D) forward(), zero_grad(), backward(), step()
    E) step(), forward(), backward(), zero_grad()

17. What does "epoch" mean in machine learning training?
    A) One forward pass through a single batch
    B) One complete pass through the entire training dataset
    C) One parameter update
    D) One validation evaluation
    E) One gradient calculation

18. Which optimizer is generally recommended as a good default choice?
    A) SGD
    B) Adam
    C) RMSprop
    D) Adagrad
    E) LBFGS

19. What is gradient clipping used for?
    A) To speed up training
    B) To prevent exploding gradients
    C) To reduce memory usage
    D) To improve accuracy
    E) To prevent overfitting

20. When implementing gradient accumulation, how should you modify the loss?
    A) Multiply by accumulation steps
    B) Divide by accumulation steps
    C) Add accumulation steps
    D) Subtract accumulation steps
    E) Leave loss unchanged

---

## Section C: Code Completion (25 points, 5 points each)

**Answer Format: Complete the missing code with proper Python syntax**

21. Complete the basic training loop:
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Clear gradients
        ________________

        # Forward pass
        outputs = model(batch.inputs)
        loss = ________________

        # Backward pass
        ________________
        ________________
        ________________
```

22. Complete the fine-tuning setup:
```python
# Load pre-trained model
model = ________________
# Replace final layer for your task
model.classifier = ________________
# Setup optimizer with lower learning rate for fine-tuning
optimizer = ________________
```

23. Complete the overfitting detection logic:
```python
def detect_overfitting(train_losses, val_losses, patience=5):
    if len(train_losses) < patience + 1:
        return False

    recent_train = ________________
    recent_val = ________________

    train_trend = ________________
    val_trend = ________________

    # Overfitting if train loss decreasing but val loss increasing
    return ________________
```

24. Complete the gradient accumulation implementation:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = ________________  # Scale loss
    loss.backward()

    if ________________:  # Check if time to update
        optimizer.step()
        ________________
```

25. Complete the learning rate scheduler setup:
```python
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

total_steps = ________________
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=________________,
    num_training_steps=________________
)
```

---

## Section D: Free Response (25 points, 5 points each)

**Answer Format: Detailed explanations with examples**

26. **Scenario Analysis (5 points)**
You're a senior developer who has been using BERT through the Transformers library for text classification via a simple pipeline. Your company now wants you to fine-tune BERT on proprietary customer support tickets to classify them into 5 categories. Explain the key differences in your approach and what new considerations you need to account for.

**Required in your answer:**
- At least 3 specific differences from using pre-trained models
- 2 potential challenges you might face
- 1 concrete recommendation for getting started

27. **Debugging Training Issues (5 points)**
Your model's training loss has been stuck at 2.3 for the last 10 epochs, while validation loss oscillates between 2.8 and 3.2. What are three possible causes and corresponding solutions?

**Required format:**
- Cause 1: [explanation] → Solution: [specific action]
- Cause 2: [explanation] → Solution: [specific action]
- Cause 3: [explanation] → Solution: [specific action]

28. **Architecture Decision (5 points)**
You need to build a system that classifies customer emails into urgent/normal categories. You have 50,000 labeled examples. Should you use fine-tuning or training from scratch? Justify your choice with at least 3 technical reasons.

**Required in your answer:**
- Clear recommendation (fine-tuning or from scratch)
- 3 technical justifications
- 1 potential drawback of your choice

29. **Data Quality Assessment (5 points)**
You receive a dataset with the following characteristics:
- 10,000 training examples
- Class distribution: 8,000 "positive", 1,500 "negative", 500 "neutral"
- Average text length: 15 words, but some texts have 200+ words
- 5% of examples have missing labels

Identify three data quality issues and propose specific solutions for each.

**Required format:**
- Issue 1: [problem] → Solution: [specific technical approach]
- Issue 2: [problem] → Solution: [specific technical approach]
- Issue 3: [problem] → Solution: [specific technical approach]

30. **Production Considerations (5 points)**
As a senior developer, what are the key differences between training a model for experimentation versus training for a production system that will serve 1M+ requests per day? List and explain 3 critical considerations.

**Required in your answer:**
- 3 production-specific considerations
- Why each matters for high-scale systems
- 1 concrete implementation recommendation for each

---

## Scoring Rubric

### True/False (20 points)
- 2 points per correct answer
- 0 points for incorrect answers

### Multiple Choice (30 points)
- 3 points per correct answer
- 0 points for incorrect answers

### Code Completion (25 points)
- 5 points per question:
  - 5 points: Completely correct and follows best practices
  - 3 points: Mostly correct with minor syntax issues
  - 1 point: Shows understanding but has significant errors
  - 0 points: Incorrect or missing

### Free Response (25 points)
- 5 points per question:
  - 5 points: Complete, accurate, demonstrates deep understanding
  - 4 points: Good understanding, minor gaps in explanation
  - 3 points: Basic understanding, missing some key points
  - 2 points: Shows some knowledge but significant gaps
  - 1 point: Minimal understanding demonstrated
  - 0 points: Incorrect or missing

### Grade Scale
- A: 90-100 points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points