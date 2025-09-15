# Evaluation and Optimization

## Model Evaluation Framework

### Beyond Accuracy: Comprehensive Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive classification evaluation"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': {
                'precision': dict(zip(self.class_names, precision_per_class)),
                'recall': dict(zip(self.class_names, recall_per_class)),
                'f1': dict(zip(self.class_names, f1_per_class))
            }
        }

        # Add probability-based metrics if available
        if y_proba is not None:
            if len(self.class_names) == 2:  # Binary classification
                results['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
                results['auc_pr'] = average_precision_score(y_true, y_proba[:, 1])
            else:  # Multi-class
                results['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')

        return results

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_errors(self, texts: List[str], y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> pd.DataFrame:
        """Analyze misclassified examples"""
        misclassified = y_true != y_pred

        error_df = pd.DataFrame({
            'text': [texts[i] for i in range(len(texts)) if misclassified[i]],
            'true_label': [self.class_names[y_true[i]] for i in range(len(y_true)) if misclassified[i]],
            'pred_label': [self.class_names[y_pred[i]] for i in range(len(y_pred)) if misclassified[i]],
        })

        if y_proba is not None:
            error_df['confidence'] = [y_proba[i].max() for i in range(len(y_proba)) if misclassified[i]]
            error_df = error_df.sort_values('confidence', ascending=False)

        return error_df

# Usage example
evaluator = ModelEvaluator(['positive', 'negative', 'neutral'])
results = evaluator.evaluate_classification(y_true, y_pred, y_proba)
print(f"Model F1 Score: {results['f1']:.4f}")
```

### Domain-Specific Evaluation

#### Text Classification Evaluation
```python
def evaluate_text_classifier(model, dataloader, device, class_names):
    """Evaluate text classification model"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Decode texts for error analysis (if needed)
            tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            all_texts.extend(texts)

    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'texts': all_texts
    }
```

#### Named Entity Recognition Evaluation
```python
from seqeval.metrics import accuracy_score, classification_report, f1_score

def evaluate_ner_model(model, dataloader, device, label_map):
    """Evaluate NER model using seqeval metrics"""
    model.eval()
    true_labels = []
    pred_labels = []

    id_to_label = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Convert to label sequences
            for i in range(len(predictions)):
                true_seq = []
                pred_seq = []

                for j in range(len(labels[i])):
                    if attention_mask[i][j] == 1 and labels[i][j] != -100:
                        true_seq.append(id_to_label[labels[i][j].item()])
                        pred_seq.append(id_to_label[predictions[i][j].item()])

                true_labels.append(true_seq)
                pred_labels.append(pred_seq)

    return {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
        'report': classification_report(true_labels, pred_labels, digits=4)
    }
```

## Hyperparameter Optimization

### Grid Search vs Random Search vs Bayesian Optimization

```python
from sklearn.model_selection import ParameterGrid
import optuna
import random

class HyperparameterOptimizer:
    def __init__(self, train_func, eval_func, config_template):
        self.train_func = train_func
        self.eval_func = eval_func
        self.config_template = config_template

    def grid_search(self, param_grid: Dict[str, List], max_trials: int = None):
        """Exhaustive grid search"""
        grid = list(ParameterGrid(param_grid))
        if max_trials:
            grid = random.sample(grid, min(max_trials, len(grid)))

        best_score = 0
        best_params = None
        results = []

        for params in tqdm(grid, desc="Grid Search"):
            # Update config with current parameters
            config = self._update_config(self.config_template, params)

            # Train and evaluate
            model_path = self.train_func(config)
            score = self.eval_func(model_path)

            results.append({'params': params, 'score': score})

            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score, results

    def bayesian_optimization(self, search_space: Dict, n_trials: int = 50):
        """Bayesian optimization using Optuna"""

        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Update config and train
            config = self._update_config(self.config_template, params)
            model_path = self.train_func(config)
            score = self.eval_func(model_path)

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value, study

    def _update_config(self, config, params):
        """Update config with new parameters"""
        new_config = copy.deepcopy(config)
        for key, value in params.items():
            # Handle nested keys like 'training.learning_rate'
            keys = key.split('.')
            obj = new_config
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)
        return new_config

# Usage example
search_space = {
    'training.learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
    'training.batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
    'model.dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
}

optimizer = HyperparameterOptimizer(train_function, eval_function, base_config)
best_params, best_score, study = optimizer.bayesian_optimization(search_space, n_trials=50)
```

### Learning Rate Finding

```python
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def find_lr(self, dataloader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Find optimal learning rate using the LR range test"""

        # Store original state
        original_state = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
        }

        # Setup
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr

        avg_loss = 0
        best_loss = float('inf')
        losses = []
        lrs = []

        data_iter = iter(dataloader)

        for i in range(num_iter):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss

            # Compute smoothed loss
            avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
            smoothed_loss = avg_loss / (1 - 0.98 ** (i + 1))

            # Stop if loss is exploding
            if smoothed_loss > 4 * best_loss:
                break

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Store values
            losses.append(smoothed_loss)
            lrs.append(lr)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = lr

        # Restore original state
        self.model.load_state_dict(original_state['model'])
        self.optimizer.load_state_dict(original_state['optimizer'])

        return lrs, losses

    def plot_lr_finder(self, lrs, losses, save_path=None):
        """Plot learning rate finder results"""
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Suggest optimal LR (steepest descent)
        min_grad_idx = np.gradient(losses).argmin()
        suggested_lr = lrs[min_grad_idx]
        print(f"Suggested learning rate: {suggested_lr:.2e}")

        return suggested_lr

# Usage
lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.find_lr(train_loader)
optimal_lr = lr_finder.plot_lr_finder(lrs, losses)
```

## Model Optimization Techniques

### Gradient Accumulation and Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

class OptimizedTrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scaler = GradScaler() if torch.cuda.is_available() else None

    def train_step_optimized(self, batch, accumulation_steps=4):
        """Optimized training step with gradient accumulation and mixed precision"""

        # Mixed precision training
        with autocast(enabled=self.scaler is not None):
            outputs = self.model(**batch)
            loss = outputs.loss / accumulation_steps  # Scale loss for accumulation

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * accumulation_steps  # Return unscaled loss for logging

    def optimizer_step_optimized(self, accumulation_steps=4, max_grad_norm=1.0):
        """Optimized optimizer step"""

        if self.scaler:
            # Gradient clipping with mixed precision
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular gradient clipping and step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()
```

### Model Pruning and Quantization

```python
import torch.nn.utils.prune as prune

def structured_pruning(model, pruning_ratio=0.2):
    """Apply structured pruning to reduce model size"""

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Prune 20% of weights in each linear layer
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)

    return model

def quantize_model(model, train_loader, device):
    """Apply dynamic quantization for inference speedup"""

    # Prepare model for quantization-aware training
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare_qat(model, inplace=False)

    # Fine-tune with quantization
    model_prepared.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model_prepared(**batch)
        # ... training code ...

    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)

    return model_quantized
```

### Early Stopping and Learning Rate Scheduling

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_score = None
        self.best_weights = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        elif val_score < self.best_score + self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())

        return False

# Advanced learning rate scheduling
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

def setup_advanced_scheduler(optimizer, scheduler_type='cosine_warm_restarts'):
    """Setup advanced learning rate scheduler"""

    if scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    elif scheduler_type == 'cosine_warm_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of epochs for first restart
            T_mult=2,  # Factor to increase T_0 after restart
            eta_min=1e-6
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
```

## Advanced Evaluation Techniques

### Cross-Validation for Robust Evaluation

```python
from sklearn.model_selection import StratifiedKFold

class CrossValidator:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

    def cross_validate(self, data, labels, train_func, eval_func):
        """Perform stratified k-fold cross-validation"""

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
            print(f"Training fold {fold + 1}/{self.n_folds}")

            # Split data
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            # Train model
            model_path = train_func(train_data, val_data, fold)

            # Evaluate
            score = eval_func(model_path, val_data)
            fold_scores.append(score)

            print(f"Fold {fold + 1} score: {score:.4f}")

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(f"Cross-validation results:")
        print(f"Mean score: {mean_score:.4f} (+/- {std_score * 2:.4f})")

        return fold_scores, mean_score, std_score
```

### Statistical Significance Testing

```python
from scipy import stats
import numpy as np

def mcnemar_test(model1_predictions, model2_predictions, true_labels):
    """McNemar's test for comparing two models"""

    # Create contingency table
    model1_correct = model1_predictions == true_labels
    model2_correct = model2_predictions == true_labels

    # Cases where models disagree
    model1_right_model2_wrong = model1_correct & ~model2_correct
    model1_wrong_model2_right = ~model1_correct & model2_correct

    b = np.sum(model1_right_model2_wrong)
    c = np.sum(model1_wrong_model2_right)

    # McNemar's test statistic
    if b + c < 25:
        # Use exact test for small samples
        p_value = stats.binom.test(b, b + c, 0.5, alternative='two-sided')
    else:
        # Use chi-square approximation
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, 1)

    return p_value, b, c

def bootstrap_confidence_interval(scores, confidence=0.95, n_bootstraps=1000):
    """Calculate bootstrap confidence interval for scores"""

    bootstrap_scores = []
    n_samples = len(scores)

    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_scores.append(np.mean(bootstrap_sample))

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    return ci_lower, ci_upper, bootstrap_scores
```

## Performance Monitoring and Debugging

### Training Monitoring Dashboard

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

class TrainingMonitor:
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics_history = []

    def update_plot(self, frame):
        """Update monitoring plots in real-time"""
        try:
            with open(self.log_file, 'r') as f:
                self.metrics_history = [json.loads(line) for line in f]
        except:
            return

        if not self.metrics_history:
            return

        epochs = [m['epoch'] for m in self.metrics_history]
        train_losses = [m['train_loss'] for m in self.metrics_history]
        val_losses = [m.get('val_loss', 0) for m in self.metrics_history]
        val_f1s = [m.get('val_f1', 0) for m in self.metrics_history]

        plt.clf()

        # Plot 1: Losses
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, val_losses, label='Val Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot 2: F1 Score
        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_f1s, label='Val F1', color='green')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        # Plot 3: Learning Rate (if available)
        if 'learning_rate' in self.metrics_history[0]:
            lrs = [m['learning_rate'] for m in self.metrics_history]
            plt.subplot(2, 2, 3)
            plt.plot(epochs, lrs, label='Learning Rate', color='orange')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

    def start_monitoring(self):
        """Start real-time monitoring"""
        fig = plt.figure(figsize=(12, 8))
        ani = FuncAnimation(fig, self.update_plot, interval=1000)
        plt.show()
        return ani
```

### Model Debugging Tools

```python
def analyze_gradients(model, dataloader, device):
    """Analyze gradient flow during training"""
    model.train()
    gradient_stats = {}

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Collect gradient statistics
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in gradient_stats:
                    gradient_stats[name] = []

                grad_norm = param.grad.norm().item()
                gradient_stats[name].append(grad_norm)

        break  # Analyze first batch only

    # Print gradient statistics
    for name, grads in gradient_stats.items():
        print(f"{name}: mean={np.mean(grads):.6f}, std={np.std(grads):.6f}")

def check_activations(model, sample_input, device):
    """Check activation patterns to detect vanishing/exploding gradients"""
    model.eval()
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        sample_input = {k: v.to(device) for k, v in sample_input.items()}
        _ = model(**sample_input)

    # Analyze activations
    for name, activation in activations.items():
        mean_act = activation.mean().item()
        std_act = activation.std().item()
        print(f"{name}: mean={mean_act:.4f}, std={std_act:.4f}")

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    return activations
```

## Common Optimization Pitfalls

### 1. Overfitting Detection
```python
def detect_overfitting(train_losses, val_losses, patience=5):
    """Detect if model is overfitting"""
    if len(train_losses) < patience + 1:
        return False

    # Check if validation loss is increasing while training loss decreases
    recent_train = train_losses[-patience:]
    recent_val = val_losses[-patience:]

    train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
    val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]

    # Overfitting if train loss decreasing but val loss increasing
    return train_trend < 0 and val_trend > 0
```

### 2. Learning Rate Too High/Low Detection
```python
def diagnose_learning_rate(losses, threshold_high=0.1, threshold_low=0.001):
    """Diagnose learning rate issues"""
    if len(losses) < 10:
        return "Need more data"

    recent_losses = losses[-10:]
    loss_std = np.std(recent_losses)
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

    if loss_std > threshold_high:
        return "Learning rate too high - loss oscillating"
    elif abs(loss_trend) < threshold_low:
        return "Learning rate too low - loss plateaued"
    else:
        return "Learning rate appears appropriate"
```

## Next Steps

Continue to [Production Deployment](05-production-deployment.md) to learn how to deploy your trained models in production environments.