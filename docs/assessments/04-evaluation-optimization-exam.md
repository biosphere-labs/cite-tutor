# Assessment 4: Evaluation and Optimization
**Time Limit: 55 minutes**
**Total Points: 100**

## Instructions
- Answer all questions in the format specified
- For True/False: Write "TRUE" or "FALSE"
- For Multiple Choice: Write the letter (A, B, C, D, E)
- For Free Response: Provide detailed explanations with code examples
- For Analysis: Show calculations and reasoning

---

## Section A: True/False (20 points, 2 points each)

**Answer Format: TRUE or FALSE**

1. Accuracy is always the best metric for evaluating classification models.

2. F1-score is the harmonic mean of precision and recall.

3. A confusion matrix can only be used for binary classification problems.

4. Cross-validation should always use the same data splits for fair comparison across experiments.

5. Learning rate scheduling always improves model performance.

6. McNemar's test can be used to compare the statistical significance of two models' performance.

7. Early stopping should always use validation loss as the stopping criterion.

8. Bayesian optimization is more efficient than grid search for hyperparameter tuning.

9. Model pruning always reduces inference time proportional to the reduction in parameters.

10. Bootstrap confidence intervals require the data to be normally distributed.

---

## Section B: Multiple Choice (25 points, 2.5 points each)

**Answer Format: Letter (A, B, C, D, or E)**

11. For a highly imbalanced dataset (95% negative, 5% positive), which metric is most informative?
    A) Accuracy
    B) Precision
    C) Recall
    D) F1-score
    E) AUC-ROC

12. What does a learning rate that's too high typically cause?
    A) Slow convergence
    B) Loss oscillation or divergence
    C) Overfitting
    D) Underfitting
    E) Memory overflow

13. In k-fold cross-validation, what happens if k equals the number of training samples?
    A) It becomes bootstrap sampling
    B) It becomes leave-one-out cross-validation
    C) It's invalid and will cause an error
    D) It becomes regular validation
    E) It becomes stratified sampling

14. Which hyperparameter optimization method is most suitable when you have expensive model training?
    A) Grid search
    B) Random search
    C) Bayesian optimization
    D) Manual tuning
    E) Evolutionary algorithms

15. What is the primary purpose of learning rate warm-up?
    A) To speed up training
    B) To prevent early unstable training
    C) To reduce memory usage
    D) To improve final accuracy
    E) To enable larger batch sizes

16. For time-series data, which cross-validation approach is most appropriate?
    A) Standard k-fold
    B) Stratified k-fold
    C) Time series split (walk-forward)
    D) Leave-one-out
    E) Bootstrap sampling

17. What does AUC-ROC measure?
    A) The area under the precision-recall curve
    B) The area under the receiver operating characteristic curve
    C) The average of precision and recall
    D) The harmonic mean of true positive and true negative rates
    E) The geometric mean of sensitivity and specificity

18. When comparing two models using McNemar's test, what does a p-value < 0.05 indicate?
    A) The models have the same performance
    B) The difference in performance is statistically significant
    C) One model is definitely better
    D) The test is invalid
    E) More data is needed

19. In the learning rate range test, what indicates the optimal learning rate?
    A) Where loss is minimum
    B) Where loss starts to decrease most rapidly
    C) Where loss becomes stable
    D) The maximum learning rate that doesn't cause divergence
    E) Where the gradient is steepest

20. What is the main advantage of using cosine annealing for learning rate scheduling?
    A) It's computationally faster
    B) It provides periodic restarts that can escape local minima
    C) It requires no hyperparameters
    D) It works with any optimizer
    E) It always converges faster

---

## Section C: Analysis and Calculation (30 points, 6 points each)

**Answer Format: Show calculations and reasoning**

21. **Metrics Analysis (6 points)**
Given the following confusion matrix for a 3-class classification problem:

```
Predicted:    A    B    C
Actual: A    85    3    2
        B     7   76    4
        C     2    1   80
```

Calculate:
- Overall accuracy
- Per-class precision for each class
- Per-class recall for each class
- Macro-averaged F1-score

**Show all calculations and formulas used.**

22. **Cross-Validation Analysis (6 points)**
You performed 5-fold cross-validation and got the following F1-scores:
[0.85, 0.87, 0.83, 0.89, 0.86]

Calculate:
- Mean F1-score
- Standard deviation
- 95% confidence interval using bootstrap method (describe approach)
- Whether this model significantly outperforms a baseline of 0.80 F1-score

**Show statistical reasoning and interpretation.**

23. **Learning Rate Optimization (6 points)**
You're implementing learning rate scheduling for a model that trains for 100 epochs with the following requirements:
- Start with learning rate of 1e-4
- Warm up for first 10 epochs to the starting LR
- Apply cosine annealing for remaining epochs
- End with learning rate of 1e-6

**Provide:**
- Mathematical formula for the warm-up phase
- Mathematical formula for the cosine annealing phase
- Learning rates at epochs: 5, 10, 50, 100

24. **Model Comparison Analysis (6 points)**
Two models performed as follows on the same test set of 1000 samples:

Model A: 920 correct predictions
Model B: 935 correct predictions

Using McNemar's test framework:
- Model A was right and Model B was wrong: 25 cases
- Model A was wrong and Model B was right: 40 cases

**Calculate:**
- The test statistic for McNemar's test
- Whether the difference is statistically significant (α = 0.05)
- Practical interpretation of the results

25. **Hyperparameter Optimization Analysis (6 points)**
You're optimizing two hyperparameters using Bayesian optimization:
- Learning rate (1e-5 to 1e-2, log scale)
- Dropout rate (0.0 to 0.5, linear scale)

After 10 trials, your best result is: LR=3e-4, Dropout=0.2, F1=0.87

**Design the next 5 trials:**
- Explain your exploration vs exploitation strategy
- Suggest specific parameter combinations to try
- Justify your choices based on Bayesian optimization principles

---

## Section D: Code Implementation (25 points, 5 points each)

**Answer Format: Complete, executable code**

26. **Comprehensive Evaluation System (5 points)**
Complete the evaluation class that calculates multiple metrics:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

class ModelEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate(self, y_true, y_pred, y_proba=None):
        """Comprehensive evaluation with multiple metrics"""

        # Basic metrics
        accuracy = ________________
        precision, recall, f1, _ = ________________

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = ________________

        # Confusion matrix
        cm = ________________

        results = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'confusion_matrix': cm.tolist(),
            'per_class': {
                class_name: {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1_score
                }
                for class_name, prec, rec, f1_score in zip(
                    self.class_names, precision_per_class, recall_per_class, f1_per_class
                )
            }
        }

        # Add AUC if probabilities provided
        if y_proba is not None:
            if len(self.class_names) == 2:
                from sklearn.metrics import roc_auc_score
                results['auc_roc'] = ________________
            else:
                from sklearn.metrics import roc_auc_score
                results['auc_roc_macro'] = ________________

        return results
```

27. **Learning Rate Finder (5 points)**
Complete the learning rate finder implementation:

```python
class LRFinder:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.original_state = None

    def find_lr(self, dataloader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Find optimal learning rate using range test"""

        # Store original state
        self.original_state = {
            'model': ________________,
            'optimizer': ________________
        }

        lr_mult = ________________
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr

        losses = []
        lrs = []
        best_loss = float('inf')

        for i in range(num_iter):
            # Get batch (cycle through dataloader if needed)
            try:
                batch = next(iter(dataloader))
            except:
                dataloader = iter(dataloader)
                batch = next(dataloader)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = ________________
            loss = ________________

            # Exponential moving average of loss
            if i == 0:
                avg_loss = loss.item()
            else:
                avg_loss = ________________  # 0.98 * avg_loss + 0.02 * loss.item()

            # Stop if loss explodes
            if avg_loss > 4 * best_loss or torch.isnan(loss):
                break

            if avg_loss < best_loss:
                best_loss = avg_loss

            # Store values
            losses.append(avg_loss)
            lrs.append(lr)

            # Backward pass
            ________________
            ________________

            # Update learning rate
            lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = lr

        # Restore original state
        ________________
        ________________

        return lrs, losses
```

28. **Cross-Validation Implementation (5 points)**
Complete the cross-validation with stratified splits:

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(model_class, X, y, cv_folds=5, **model_kwargs):
    """Perform stratified k-fold cross-validation"""

    skf = ________________
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(________________):
        print(f"Training fold {fold + 1}/{cv_folds}")

        # Split data
        X_train, X_val = ________________, ________________
        y_train, y_val = ________________, ________________

        # Initialize fresh model
        model = ________________

        # Train model (implement your training logic)
        trained_model = train_model(model, X_train, y_train)

        # Evaluate
        predictions = ________________
        fold_score = ________________  # Calculate F1 score

        fold_results.append({
            'fold': fold,
            'score': fold_score,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        })

    # Calculate statistics
    scores = [result['score'] for result in fold_results]
    cv_results = {
        'mean_score': ________________,
        'std_score': ________________,
        'scores': scores,
        'fold_details': fold_results
    }

    return cv_results
```

29. **Bayesian Optimization Setup (5 points)**
Complete the Bayesian optimization for hyperparameter tuning:

```python
import optuna

class BayesianOptimizer:
    def __init__(self, train_func, eval_func):
        self.train_func = train_func
        self.eval_func = eval_func

    def optimize(self, n_trials=50):
        """Optimize hyperparameters using Bayesian optimization"""

        def objective(trial):
            # Sample hyperparameters
            lr = ________________  # Log scale between 1e-5 and 1e-2
            batch_size = ________________  # Categorical: [16, 32, 64, 128]
            dropout = ________________  # Float between 0.1 and 0.5
            weight_decay = ________________  # Log scale between 1e-6 and 1e-2

            # Create config
            config = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'dropout': dropout,
                'weight_decay': weight_decay
            }

            # Train and evaluate model
            model_path = self.train_func(config)
            score = self.eval_func(model_path)

            return score

        # Create study and optimize
        study = ________________
        study.optimize(objective, n_trials=n_trials)

        return {
            'best_params': ________________,
            'best_score': ________________,
            'study': study
        }
```

30. **Statistical Significance Testing (5 points)**
Complete the model comparison with statistical tests:

```python
from scipy import stats
import numpy as np

def compare_models_statistically(model1_predictions, model2_predictions, true_labels):
    """Compare two models using statistical tests"""

    # Calculate accuracies
    acc1 = ________________
    acc2 = ________________

    # McNemar's test for dependent samples
    # Create contingency table
    model1_correct = ________________
    model2_correct = ________________

    # Cases where models disagree
    model1_right_model2_wrong = ________________
    model1_wrong_model2_right = ________________

    b = np.sum(model1_right_model2_wrong)
    c = np.sum(model1_wrong_model2_right)

    # McNemar's test statistic
    if b + c < 25:
        # Use exact binomial test for small samples
        p_value = ________________
    else:
        # Use chi-square approximation
        chi2_stat = ________________
        p_value = ________________

    # Bootstrap confidence interval for difference
    def bootstrap_diff(n_bootstrap=1000):
        differences = []
        n_samples = len(true_labels)

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = ________________
            sampled_true = true_labels[indices]
            sampled_pred1 = model1_predictions[indices]
            sampled_pred2 = model2_predictions[indices]

            # Calculate difference in accuracy
            acc_diff = ________________
            differences.append(acc_diff)

        return np.array(differences)

    bootstrap_diffs = bootstrap_diff()
    ci_lower, ci_upper = ________________  # 2.5% and 97.5% percentiles

    return {
        'model1_accuracy': acc1,
        'model2_accuracy': acc2,
        'accuracy_difference': acc2 - acc1,
        'mcnemar_p_value': p_value,
        'significant_difference': p_value < 0.05,
        'confidence_interval_95': (ci_lower, ci_upper),
        'disagreement_cases': {'b': b, 'c': c}
    }
```

---

## Scoring Rubric

### True/False (20 points)
- 2 points per correct answer

### Multiple Choice (25 points)
- 2.5 points per correct answer

### Analysis and Calculation (30 points)
- 6 points per question:
  - 6 points: Correct calculations with clear methodology
  - 5 points: Mostly correct with minor errors
  - 4 points: Good understanding, some calculation errors
  - 3 points: Basic understanding, significant errors
  - 2 points: Partial understanding
  - 0-1 points: Incorrect or missing

### Code Implementation (25 points)
- 5 points per question:
  - 5 points: Completely correct and efficient
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