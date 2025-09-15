# Debugging and Monitoring AI Systems

## Debugging Training Issues

### Common Training Problems and Solutions

#### 1. Loss Not Decreasing
```python
import matplotlib.pyplot as plt
import numpy as np

class TrainingDiagnostics:
    def __init__(self):
        self.metrics_history = []

    def diagnose_training_stagnation(self, losses: List[float], patience: int = 10) -> Dict[str, Any]:
        """Diagnose why training loss isn't improving"""

        if len(losses) < patience:
            return {"status": "insufficient_data", "message": "Need more epochs to diagnose"}

        recent_losses = losses[-patience:]

        # Check for oscillation
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        cv = loss_std / loss_mean if loss_mean > 0 else float('inf')

        # Check for trend
        x = np.arange(len(recent_losses))
        slope, _ = np.polyfit(x, recent_losses, 1)

        diagnosis = {
            "coefficient_of_variation": cv,
            "trend_slope": slope,
            "recent_mean_loss": loss_mean,
            "recent_std_loss": loss_std
        }

        # Provide recommendations
        if cv > 0.1:  # High oscillation
            diagnosis["issue"] = "high_oscillation"
            diagnosis["recommendation"] = "Reduce learning rate by factor of 2-5"
        elif abs(slope) < 1e-6:  # Plateau
            diagnosis["issue"] = "plateau"
            diagnosis["recommendation"] = "Try learning rate scheduler or different optimizer"
        elif slope > 0:  # Increasing loss
            diagnosis["issue"] = "diverging"
            diagnosis["recommendation"] = "Significantly reduce learning rate or check data"
        else:
            diagnosis["issue"] = "normal_convergence"
            diagnosis["recommendation"] = "Training appears to be converging normally"

        return diagnosis

    def check_gradient_flow(self, model: torch.nn.Module) -> Dict[str, float]:
        """Check if gradients are flowing properly through the model"""

        gradient_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats[name] = {
                    'gradient_norm': grad_norm,
                    'param_norm': param.norm().item(),
                    'grad_to_param_ratio': grad_norm / (param.norm().item() + 1e-8)
                }

        # Check for vanishing/exploding gradients
        grad_norms = [stats['gradient_norm'] for stats in gradient_stats.values()]

        analysis = {
            'mean_grad_norm': np.mean(grad_norms),
            'max_grad_norm': np.max(grad_norms),
            'min_grad_norm': np.min(grad_norms),
            'gradients_stats': gradient_stats
        }

        if analysis['max_grad_norm'] > 100:
            analysis['issue'] = 'exploding_gradients'
            analysis['recommendation'] = 'Use gradient clipping or reduce learning rate'
        elif analysis['mean_grad_norm'] < 1e-7:
            analysis['issue'] = 'vanishing_gradients'
            analysis['recommendation'] = 'Check model initialization or use skip connections'
        else:
            analysis['issue'] = 'normal'

        return analysis

    def analyze_data_distribution(self, dataloader, tokenizer=None) -> Dict[str, Any]:
        """Analyze data distribution to identify potential issues"""

        text_lengths = []
        label_counts = defaultdict(int)

        for batch in dataloader:
            if 'input_ids' in batch:
                # For transformer models
                lengths = (batch['input_ids'] != tokenizer.pad_token_id).sum(dim=1)
                text_lengths.extend(lengths.tolist())
            elif 'text' in batch:
                # For raw text
                text_lengths.extend([len(text.split()) for text in batch['text']])

            if 'labels' in batch:
                labels = batch['labels'].tolist()
                for label in labels:
                    label_counts[label] += 1

        analysis = {
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths),
                'percentiles': {
                    '25': np.percentile(text_lengths, 25),
                    '50': np.percentile(text_lengths, 50),
                    '75': np.percentile(text_lengths, 75),
                    '95': np.percentile(text_lengths, 95)
                }
            },
            'label_distribution': dict(label_counts),
            'class_imbalance_ratio': max(label_counts.values()) / min(label_counts.values()) if label_counts else 1
        }

        # Check for issues
        if analysis['class_imbalance_ratio'] > 10:
            analysis['issue'] = 'severe_class_imbalance'
            analysis['recommendation'] = 'Use weighted loss or resampling techniques'
        elif analysis['text_length_stats']['std'] > analysis['text_length_stats']['mean']:
            analysis['issue'] = 'high_length_variance'
            analysis['recommendation'] = 'Consider dynamic batching or length-based bucketing'

        return analysis

# Usage example
diagnostics = TrainingDiagnostics()

# During training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)

    # Diagnose training issues
    diagnosis = diagnostics.diagnose_training_stagnation(train_losses)
    if diagnosis['issue'] != 'normal_convergence':
        print(f"Training issue detected: {diagnosis['issue']}")
        print(f"Recommendation: {diagnosis['recommendation']}")

    # Check gradient flow
    grad_analysis = diagnostics.check_gradient_flow(model)
    if grad_analysis['issue'] != 'normal':
        print(f"Gradient issue: {grad_analysis['issue']}")
```

#### 2. Overfitting Detection and Prevention
```python
class OverfittingDetector:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss: float, val_loss: float):
        """Update with new loss values"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def detect_overfitting(self) -> Dict[str, Any]:
        """Detect overfitting patterns"""

        if len(self.train_losses) < self.patience + 1:
            return {"status": "insufficient_data"}

        # Calculate recent trends
        recent_train = self.train_losses[-self.patience:]
        recent_val = self.val_losses[-self.patience:]

        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]

        # Calculate gap between train and validation loss
        current_gap = self.val_losses[-1] - self.train_losses[-1]
        avg_gap = np.mean([v - t for v, t in zip(self.val_losses, self.train_losses)])

        analysis = {
            "train_trend": train_trend,
            "val_trend": val_trend,
            "current_gap": current_gap,
            "average_gap": avg_gap,
            "gap_increasing": current_gap > avg_gap + self.min_delta
        }

        # Detect overfitting
        if train_trend < -self.min_delta and val_trend > self.min_delta:
            analysis["status"] = "overfitting_detected"
            analysis["severity"] = "high" if current_gap > 2 * avg_gap else "moderate"
            analysis["recommendation"] = [
                "Reduce model complexity",
                "Increase regularization",
                "Add more training data",
                "Implement early stopping"
            ]
        elif analysis["gap_increasing"]:
            analysis["status"] = "potential_overfitting"
            analysis["recommendation"] = [
                "Monitor closely",
                "Consider early stopping",
                "Validate on separate test set"
            ]
        else:
            analysis["status"] = "healthy_training"

        return analysis

# Integration with training loop
overfitting_detector = OverfittingDetector(patience=5)

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate_epoch()

    overfitting_detector.update(train_loss, val_loss)
    overfitting_analysis = overfitting_detector.detect_overfitting()

    if overfitting_analysis["status"] == "overfitting_detected":
        print(f"Overfitting detected! Severity: {overfitting_analysis['severity']}")
        # Implement corrective actions
        break
```

### Model Debugging Tools

#### 3. Attention Visualization for Transformers
```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_attention_weights(self, text: str, layer: int = -1, head: int = 0):
        """Extract attention weights from transformer model"""

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt')

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Get attention weights
        attention = outputs.attentions[layer][0, head]  # [seq_len, seq_len]

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return attention.numpy(), tokens

    def plot_attention_heatmap(self, text: str, layer: int = -1, head: int = 0, save_path: str = None):
        """Plot attention heatmap"""

        attention, tokens = self.extract_attention_weights(text, layer, head)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            cbar=True
        )
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_attention_patterns(self, texts: List[str], layer: int = -1):
        """Analyze attention patterns across multiple examples"""

        attention_stats = {
            'self_attention_scores': [],
            'attention_entropy': [],
            'max_attention_scores': []
        }

        for text in texts:
            attention, tokens = self.extract_attention_weights(text, layer)

            # Self-attention (diagonal)
            self_attn = np.diag(attention).mean()
            attention_stats['self_attention_scores'].append(self_attn)

            # Attention entropy (how distributed attention is)
            entropy = -np.sum(attention * np.log(attention + 1e-9), axis=1).mean()
            attention_stats['attention_entropy'].append(entropy)

            # Maximum attention per token
            max_attn = attention.max(axis=1).mean()
            attention_stats['max_attention_scores'].append(max_attn)

        # Summary statistics
        summary = {}
        for key, values in attention_stats.items():
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return summary

# Usage
visualizer = AttentionVisualizer(model, tokenizer)

# Visualize attention for a specific example
visualizer.plot_attention_heatmap("The model attention is focused on important words.")

# Analyze patterns across multiple examples
attention_analysis = visualizer.analyze_attention_patterns(validation_texts)
print(f"Average attention entropy: {attention_analysis['attention_entropy']['mean']:.3f}")
```

#### 4. Feature Importance and Interpretability
```python
from captum.attr import IntegratedGradients, LayerConductance, TokenReferenceBase

class ModelInterpreter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_token_id = tokenizer.pad_token_id

    def explain_prediction(self, text: str, target_class: int = None) -> Dict[str, Any]:
        """Explain model prediction using integrated gradients"""

        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']

        # Baseline (reference)
        reference_ids = torch.zeros_like(input_ids).fill_(self.reference_token_id)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0, predicted_class].item()

        # If target class not specified, use predicted class
        if target_class is None:
            target_class = predicted_class

        # Integrated gradients
        ig = IntegratedGradients(self._forward_func)
        attributions = ig.attribute(
            input_ids,
            reference_ids,
            target=target_class,
            n_steps=50
        )

        # Convert to interpretable format
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_attributions = attributions[0].sum(dim=-1).tolist()

        # Normalize attributions
        max_attr = max(abs(attr) for attr in token_attributions)
        normalized_attributions = [attr / max_attr for attr in token_attributions]

        return {
            'text': text,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'target_class': target_class,
            'tokens': tokens,
            'attributions': normalized_attributions,
            'token_importance': list(zip(tokens, normalized_attributions))
        }

    def _forward_func(self, input_ids):
        """Forward function for attribution methods"""
        outputs = self.model(input_ids)
        return outputs.logits

    def visualize_attributions(self, explanation: Dict, save_path: str = None):
        """Visualize token attributions"""

        tokens = explanation['tokens']
        attributions = explanation['attributions']

        # Create color map
        colors = []
        for attr in attributions:
            if attr > 0:
                colors.append(f'rgba(0, 255, 0, {abs(attr)})')  # Green for positive
            else:
                colors.append(f'rgba(255, 0, 0, {abs(attr)})')  # Red for negative

        # Plot
        fig, ax = plt.subplots(figsize=(15, 3))

        # Remove special tokens for visualization
        display_tokens = []
        display_colors = []
        display_attrs = []

        for token, color, attr in zip(tokens, colors, attributions):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                display_tokens.append(token.replace('##', ''))
                display_colors.append(color)
                display_attrs.append(attr)

        # Create bars
        bars = ax.bar(range(len(display_tokens)), display_attrs, color=display_colors)

        ax.set_xticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right')
        ax.set_ylabel('Attribution Score')
        ax.set_title(f'Token Attributions (Predicted: {explanation["predicted_class"]}, Confidence: {explanation["confidence"]:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels on bars
        for bar, attr in zip(bars, display_attrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{attr:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Usage
interpreter = ModelInterpreter(model, tokenizer)

# Explain a prediction
explanation = interpreter.explain_prediction("This movie is absolutely terrible and boring.")
interpreter.visualize_attributions(explanation)

# Print top contributing tokens
sorted_tokens = sorted(explanation['token_importance'], key=lambda x: abs(x[1]), reverse=True)
print("Top contributing tokens:")
for token, importance in sorted_tokens[:5]:
    print(f"{token}: {importance:.3f}")
```

## Production Monitoring

### Real-time Performance Monitoring

```python
import time
import asyncio
from collections import deque, defaultdict
import numpy as np

class ModelPerformanceMonitor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = {
            'latency': deque(maxlen=window_size),
            'confidence_scores': deque(maxlen=window_size),
            'error_count': 0,
            'total_requests': 0,
            'predictions_per_class': defaultdict(int)
        }
        self.alerts = []

    def record_prediction(self, latency: float, confidence: float, prediction: int, error: bool = False):
        """Record a prediction for monitoring"""

        self.metrics['latency'].append(latency)
        self.metrics['confidence_scores'].append(confidence)
        self.metrics['predictions_per_class'][prediction] += 1
        self.metrics['total_requests'] += 1

        if error:
            self.metrics['error_count'] += 1

        # Check for alerts
        self._check_alerts()

    def _check_alerts(self):
        """Check for performance alerts"""

        if len(self.metrics['latency']) < 100:  # Need sufficient data
            return

        # High latency alert
        recent_latency = list(self.metrics['latency'])[-100:]
        avg_latency = np.mean(recent_latency)
        p95_latency = np.percentile(recent_latency, 95)

        if avg_latency > 1.0:  # 1 second threshold
            self.alerts.append({
                'type': 'high_latency',
                'message': f'Average latency high: {avg_latency:.3f}s',
                'timestamp': time.time(),
                'value': avg_latency
            })

        # Low confidence alert
        recent_confidence = list(self.metrics['confidence_scores'])[-100:]
        avg_confidence = np.mean(recent_confidence)

        if avg_confidence < 0.7:  # Low confidence threshold
            self.alerts.append({
                'type': 'low_confidence',
                'message': f'Average confidence low: {avg_confidence:.3f}',
                'timestamp': time.time(),
                'value': avg_confidence
            })

        # Error rate alert
        error_rate = self.metrics['error_count'] / self.metrics['total_requests']
        if error_rate > 0.05:  # 5% error rate
            self.alerts.append({
                'type': 'high_error_rate',
                'message': f'Error rate high: {error_rate:.1%}',
                'timestamp': time.time(),
                'value': error_rate
            })

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""

        if not self.metrics['latency']:
            return {"status": "no_data"}

        latency_data = list(self.metrics['latency'])
        confidence_data = list(self.metrics['confidence_scores'])

        return {
            'total_requests': self.metrics['total_requests'],
            'error_rate': self.metrics['error_count'] / max(self.metrics['total_requests'], 1),
            'latency': {
                'mean': np.mean(latency_data),
                'p50': np.percentile(latency_data, 50),
                'p95': np.percentile(latency_data, 95),
                'p99': np.percentile(latency_data, 99),
                'max': np.max(latency_data)
            },
            'confidence': {
                'mean': np.mean(confidence_data),
                'std': np.std(confidence_data),
                'min': np.min(confidence_data)
            },
            'predictions_distribution': dict(self.metrics['predictions_per_class']),
            'recent_alerts': self.alerts[-10:]  # Last 10 alerts
        }

# Data Drift Detection
class DataDriftDetector:
    def __init__(self, reference_data: np.ndarray, drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.current_window = deque(maxlen=1000)

    def add_sample(self, sample: np.ndarray):
        """Add new sample to current window"""
        self.current_window.append(sample)

    def detect_drift(self) -> Dict[str, Any]:
        """Detect if data distribution has drifted"""

        if len(self.current_window) < 100:
            return {"status": "insufficient_data"}

        current_data = np.array(list(self.current_window))

        # Kolmogorov-Smirnov test for each feature
        from scipy import stats

        drift_scores = []
        p_values = []

        for i in range(self.reference_data.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                current_data[:, i]
            )
            drift_scores.append(ks_stat)
            p_values.append(p_value)

        # Overall drift assessment
        significant_drifts = sum(1 for p in p_values if p < self.drift_threshold)
        drift_ratio = significant_drifts / len(p_values)

        return {
            "drift_detected": drift_ratio > 0.1,  # If more than 10% of features show drift
            "drift_ratio": drift_ratio,
            "max_drift_score": max(drift_scores),
            "avg_drift_score": np.mean(drift_scores),
            "feature_drift_scores": drift_scores,
            "p_values": p_values,
            "recommendation": "Retrain model" if drift_ratio > 0.2 else "Monitor closely"
        }

# Integration with FastAPI
monitor = ModelPerformanceMonitor()
drift_detector = DataDriftDetector(reference_embeddings)

@app.post("/predict_monitored")
async def predict_with_monitoring(request: PredictionRequest):
    start_time = time.time()

    try:
        # Make prediction
        result = model.predict([request.text])[0]

        # Calculate latency
        latency = time.time() - start_time

        # Record metrics
        monitor.record_prediction(
            latency=latency,
            confidence=result['confidence'],
            prediction=result['prediction'],
            error=False
        )

        # Check for data drift (if applicable)
        if hasattr(result, 'embeddings'):
            drift_detector.add_sample(result['embeddings'])

        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            processing_time_ms=latency * 1000,
            model_version="1.0.0"
        )

    except Exception as e:
        # Record error
        monitor.record_prediction(
            latency=time.time() - start_time,
            confidence=0.0,
            prediction=-1,
            error=True
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/metrics")
async def get_monitoring_metrics():
    """Get current monitoring metrics"""
    metrics = monitor.get_metrics_summary()
    drift_status = drift_detector.detect_drift()

    return {
        "performance_metrics": metrics,
        "data_drift": drift_status,
        "timestamp": time.time()
    }
```

### Model Health Checks

```python
class ModelHealthChecker:
    def __init__(self, model, tokenizer, test_cases: List[Dict]):
        self.model = model
        self.tokenizer = tokenizer
        self.test_cases = test_cases  # Known input-output pairs

    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive model health check"""

        results = {
            "timestamp": time.time(),
            "overall_health": "unknown",
            "checks": {}
        }

        # 1. Basic functionality test
        results["checks"]["basic_functionality"] = self._test_basic_functionality()

        # 2. Consistency test
        results["checks"]["consistency"] = self._test_consistency()

        # 3. Performance regression test
        results["checks"]["performance_regression"] = self._test_performance_regression()

        # 4. Memory usage test
        results["checks"]["memory_usage"] = self._test_memory_usage()

        # Determine overall health
        failed_checks = [name for name, result in results["checks"].items()
                        if result["status"] == "failed"]

        if not failed_checks:
            results["overall_health"] = "healthy"
        elif len(failed_checks) == 1 and failed_checks[0] == "memory_usage":
            results["overall_health"] = "warning"
        else:
            results["overall_health"] = "unhealthy"

        results["failed_checks"] = failed_checks

        return results

    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic model functionality"""
        try:
            # Test with a simple input
            test_input = "This is a test sentence."

            start_time = time.time()
            inputs = self.tokenizer(test_input, return_tensors='pt')

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)

            latency = time.time() - start_time

            # Validate output shape and values
            if outputs.logits.shape[1] != self.model.config.num_labels:
                return {
                    "status": "failed",
                    "error": f"Wrong output shape: {outputs.logits.shape}"
                }

            if torch.isnan(outputs.logits).any():
                return {
                    "status": "failed",
                    "error": "NaN values in model output"
                }

            return {
                "status": "passed",
                "latency": latency,
                "output_shape": list(outputs.logits.shape),
                "prediction_probs": predictions[0].tolist()
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def _test_consistency(self) -> Dict[str, Any]:
        """Test model consistency on known test cases"""

        try:
            inconsistencies = []

            for test_case in self.test_cases:
                inputs = self.tokenizer(test_case['input'], return_tensors='pt')

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

                if predicted_class != test_case['expected_output']:
                    inconsistencies.append({
                        'input': test_case['input'],
                        'expected': test_case['expected_output'],
                        'actual': predicted_class
                    })

            consistency_rate = 1 - len(inconsistencies) / len(self.test_cases)

            return {
                "status": "passed" if consistency_rate >= 0.9 else "failed",
                "consistency_rate": consistency_rate,
                "total_test_cases": len(self.test_cases),
                "inconsistencies": inconsistencies[:5]  # Show first 5
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def _test_performance_regression(self) -> Dict[str, Any]:
        """Test for performance regression"""

        try:
            latencies = []
            test_inputs = [case['input'] for case in self.test_cases[:10]]

            for test_input in test_inputs:
                start_time = time.time()

                inputs = self.tokenizer(test_input, return_tensors='pt')
                with torch.no_grad():
                    _ = self.model(**inputs)

                latencies.append(time.time() - start_time)

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            # Define acceptable thresholds
            max_avg_latency = 1.0  # 1 second
            max_p95_latency = 2.0  # 2 seconds

            return {
                "status": "passed" if avg_latency < max_avg_latency and p95_latency < max_p95_latency else "failed",
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "max_latency": max(latencies),
                "thresholds": {
                    "max_avg_latency": max_avg_latency,
                    "max_p95_latency": max_p95_latency
                }
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""

        try:
            import psutil
            import gc

            # Clear cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            # Measure baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run inference
            test_input = "This is a memory test with a longer sentence to check memory usage patterns."
            inputs = self.tokenizer(test_input, return_tensors='pt')

            with torch.no_grad():
                _ = self.model(**inputs)

            # Measure memory after inference
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory

            # Check GPU memory if available
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            return {
                "status": "passed" if memory_increase < 500 else "warning",  # 500MB threshold
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory,
                "memory_increase_mb": memory_increase,
                "gpu_memory_mb": gpu_memory
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

# Automated health check endpoint
health_checker = ModelHealthChecker(model, tokenizer, test_cases)

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check endpoint"""
    return health_checker.run_health_check()

# Scheduled health checks
async def scheduled_health_check():
    """Run health checks periodically"""
    while True:
        try:
            health_result = health_checker.run_health_check()

            if health_result["overall_health"] == "unhealthy":
                # Send alert (implement your alerting mechanism)
                print(f"ALERT: Model health check failed: {health_result['failed_checks']}")

            # Log health check result
            logger.info("health_check_completed", health_status=health_result["overall_health"])

        except Exception as e:
            logger.error("health_check_failed", error=str(e))

        # Wait 5 minutes before next check
        await asyncio.sleep(300)

# Start background health checking
@app.on_event("startup")
async def start_health_monitoring():
    asyncio.create_task(scheduled_health_check())
```

## Alerting and Incident Response

### Alert Management System
```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Callable
import smtplib
from email.mime.text import MIMEText

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    source: str
    resolved: bool = False
    resolution_notes: str = ""

class AlertManager:
    def __init__(self):
        self.alerts: List[Alert] = []
        self.handlers: Dict[AlertSeverity, List[Callable]] = {
            AlertSeverity.LOW: [],
            AlertSeverity.MEDIUM: [],
            AlertSeverity.HIGH: [],
            AlertSeverity.CRITICAL: []
        }

    def register_handler(self, severity: AlertSeverity, handler: Callable):
        """Register alert handler for specific severity"""
        self.handlers[severity].append(handler)

    def create_alert(self, severity: AlertSeverity, title: str, description: str, source: str) -> str:
        """Create new alert"""
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"

        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=time.time(),
            source=source
        )

        self.alerts.append(alert)

        # Trigger handlers
        for handler in self.handlers[severity]:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler failed: {e}")

        return alert_id

    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_notes = resolution_notes
                break

    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [alert for alert in self.alerts if not alert.resolved]

# Alert handlers
def email_alert_handler(alert: Alert):
    """Send email alert"""
    # Configure your email settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    email_user = "alerts@yourcompany.com"
    email_password = "your_password"
    recipients = ["devops@yourcompany.com"]

    subject = f"[{alert.severity.value.upper()}] {alert.title}"
    body = f"""
    Alert Details:
    ID: {alert.id}
    Severity: {alert.severity.value}
    Source: {alert.source}
    Time: {time.ctime(alert.timestamp)}

    Description:
    {alert.description}
    """

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = email_user
    msg['To'] = ", ".join(recipients)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_user, email_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Failed to send email alert: {e}")

def slack_alert_handler(alert: Alert):
    """Send Slack alert (pseudo-code)"""
    # Implement Slack webhook integration
    webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

    message = {
        "text": f"🚨 {alert.title}",
        "attachments": [
            {
                "color": "danger" if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] else "warning",
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Description", "value": alert.description, "short": False}
                ]
            }
        ]
    }

    # Send to Slack (implement HTTP request)
    # requests.post(webhook_url, json=message)

# Setup alert manager
alert_manager = AlertManager()
alert_manager.register_handler(AlertSeverity.HIGH, email_alert_handler)
alert_manager.register_handler(AlertSeverity.CRITICAL, email_alert_handler)
alert_manager.register_handler(AlertSeverity.CRITICAL, slack_alert_handler)

# Integration with monitoring
def check_and_alert():
    """Check metrics and create alerts if needed"""
    metrics = monitor.get_metrics_summary()

    # High error rate alert
    if metrics.get('error_rate', 0) > 0.05:
        alert_manager.create_alert(
            AlertSeverity.HIGH,
            "High Error Rate Detected",
            f"Error rate is {metrics['error_rate']:.1%}, exceeding 5% threshold",
            "model_monitor"
        )

    # High latency alert
    if metrics.get('latency', {}).get('p95', 0) > 2.0:
        alert_manager.create_alert(
            AlertSeverity.MEDIUM,
            "High Latency Detected",
            f"P95 latency is {metrics['latency']['p95']:.3f}s, exceeding 2s threshold",
            "model_monitor"
        )

    # Data drift alert
    drift_status = drift_detector.detect_drift()
    if drift_status.get('drift_detected', False):
        alert_manager.create_alert(
            AlertSeverity.MEDIUM,
            "Data Drift Detected",
            f"Data drift detected with ratio {drift_status['drift_ratio']:.1%}",
            "drift_detector"
        )

@app.get("/alerts")
async def get_alerts():
    """Get active alerts"""
    return {
        "active_alerts": [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp,
                "source": alert.source
            }
            for alert in alert_manager.get_active_alerts()
        ]
    }

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolution_notes: str = ""):
    """Resolve an alert"""
    alert_manager.resolve_alert(alert_id, resolution_notes)
    return {"message": f"Alert {alert_id} resolved"}
```

This comprehensive debugging and monitoring guide provides production-ready tools for maintaining AI systems. The code examples cover training diagnostics, model interpretation, real-time monitoring, health checks, and alerting systems that are essential for reliable AI deployments.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Write training fundamentals for experienced developers", "status": "completed", "activeForm": "Writing training fundamentals for experienced developers"}, {"content": "Write model architectures and selection guide", "status": "completed", "activeForm": "Writing model architectures and selection guide"}, {"content": "Write training pipeline and data preparation", "status": "completed", "activeForm": "Writing training pipeline and data preparation"}, {"content": "Write evaluation and optimization techniques", "status": "completed", "activeForm": "Writing evaluation and optimization techniques"}, {"content": "Write production deployment considerations", "status": "completed", "activeForm": "Writing production deployment considerations"}, {"content": "Write debugging and monitoring guide", "status": "completed", "activeForm": "Writing debugging and monitoring guide"}, {"content": "Create README with focused learning path", "status": "in_progress", "activeForm": "Creating README with focused learning path"}]