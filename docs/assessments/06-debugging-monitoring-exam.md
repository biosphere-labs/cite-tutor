# Assessment 6: Debugging and Monitoring
**Time Limit: 70 minutes**
**Total Points: 100**

## Instructions
- Answer all questions in the format specified
- For True/False: Write "TRUE" or "FALSE"
- For Multiple Choice: Write the letter (A, B, C, D, E)
- For Free Response: Provide detailed explanations with code examples
- For Debugging Scenarios: Analyze symptoms and propose solutions

---

## Section A: True/False (20 points, 2 points each)

**Answer Format: TRUE or FALSE**

1. A training loss that oscillates wildly typically indicates the learning rate is too high.

2. Attention visualizations can help debug why a transformer model makes specific predictions.

3. Data drift detection should only be performed when model performance degrades.

4. Gradient norms consistently near zero indicate the model is converging well.

5. Real-time monitoring should track both model performance metrics and system metrics.

6. A model that performs well on validation data but poorly in production always indicates overfitting.

7. Memory leaks in ML training loops typically manifest as gradually increasing GPU memory usage.

8. Alert fatigue can be reduced by implementing proper alert thresholds and aggregation.

9. Model interpretability techniques are only useful for debugging, not for production monitoring.

10. Cross-validation scores that vary significantly across folds indicate potential data quality issues.

---

## Section B: Multiple Choice (25 points, 2.5 points each)

**Answer Format: Letter (A, B, C, D, or E)**

11. Your model's training loss decreases normally, but validation loss starts increasing after epoch 5. What is the most likely cause?
    A) Learning rate is too low
    B) The model is overfitting
    C) Data preprocessing error
    D) Optimizer configuration issue
    E) Insufficient training data

12. When debugging gradient flow in a transformer model, what indicates vanishing gradients?
    A) Gradient norms > 100
    B) Gradient norms approaching zero in early layers
    C) Oscillating loss values
    D) Increasing validation accuracy
    E) Memory usage spikes

13. For production model monitoring, which combination of metrics is most critical?
    A) Accuracy and F1-score only
    B) Latency and throughput only
    C) Latency, accuracy, and data distribution
    D) CPU usage and memory consumption only
    E) Request count and error rate only

14. What is the most effective way to detect data drift in production?
    A) Monitor model accuracy daily
    B) Compare feature distributions between training and production data
    C) Track request volume changes
    D) Monitor system resource usage
    E) Check for new error types

15. When your model shows high confidence but low accuracy in production, the most likely cause is:
    A) Model calibration issues
    B) Hardware failures
    C) Network latency
    D) Database corruption
    E) Load balancer configuration

16. For debugging attention mechanisms in transformers, which visualization is most informative?
    A) Loss curves over time
    B) Weight histograms
    C) Attention heatmaps for specific examples
    D) Gradient flow diagrams
    E) Confusion matrices

17. What does a consistently high learning rate range test suggest about your model?
    A) The model architecture is optimal
    B) The loss function is inappropriate
    C) The data preprocessing needs adjustment
    D) The model may be too simple for the task
    E) The optimizer choice is incorrect

18. In production monitoring, what pattern suggests model degradation due to data drift?
    A) Sudden spike in error rate
    B) Gradual decline in accuracy over time
    C) Increased memory usage
    D) Higher request latency
    E) More frequent timeouts

19. When debugging memory issues during training, which tool provides the most actionable insights?
    A) System memory monitors
    B) GPU memory profilers (e.g., nvidia-smi)
    C) Network traffic analyzers
    D) Disk usage monitors
    E) CPU profilers

20. For alerting systems in production ML, what threshold strategy minimizes false positives?
    A) Fixed thresholds for all metrics
    B) Adaptive thresholds based on historical data
    C) Manual threshold adjustment
    D) No thresholds, alert on any change
    E) Percentage-based thresholds only

---

## Section C: Debugging Scenarios (30 points, 6 points each)

**Answer Format: Analyze symptoms, identify root causes, propose specific solutions**

21. **Training Stagnation Scenario (6 points)**
You're training a BERT-based text classifier. The symptoms are:
- Training loss stuck at 1.2 for the last 15 epochs
- Validation loss oscillating between 1.15 and 1.25
- Model accuracy plateau at 65%
- Learning rate: 2e-5, Batch size: 32, Dataset: 10K examples

**Analyze and provide:**
- 2 most likely root causes
- Specific diagnostic steps to confirm each cause
- Concrete solutions with implementation details
- How to prevent this issue in future training

22. **Production Performance Degradation (6 points)**
Your production sentiment analysis model shows these symptoms over the past week:
- Accuracy dropped from 87% to 78%
- Average confidence scores decreased from 0.85 to 0.72
- Request latency increased by 30%
- No changes to model or infrastructure

**Required analysis:**
- 3 potential causes of this degradation
- Monitoring data you would examine to diagnose each cause
- Step-by-step investigation plan
- Remediation strategies for each potential cause

23. **Memory and Performance Issues (6 points)**
During training, you experience:
- GPU memory gradually increases from 6GB to 11GB over 50 epochs
- Training speed decreases from 30 sec/epoch to 2 min/epoch
- Eventually getting CUDA out-of-memory errors
- Model: BERT-large, Batch size: 16, Sequence length: 512

**Provide detailed analysis:**
- Root cause identification with technical explanation
- Diagnostic commands/tools to confirm the issue
- Multiple solution approaches with trade-offs
- Code changes needed to implement the best solution

24. **Model Interpretability Debugging (6 points)**
Your NER model has these issues:
- F1-score is 0.78 overall but varies significantly by entity type
- Person names: F1=0.89, Organizations: F1=0.52, Locations: F1=0.81
- Attention seems unfocused when visualized
- Training data: 50K examples with class imbalance

**Required debugging approach:**
- Specific analysis techniques to understand the poor organization detection
- How to use attention visualization for diagnosis
- Data quality checks to perform
- Model architecture modifications to consider

25. **Alert Storm Investigation (6 points)**
Your production ML system is generating excessive alerts:
- 200+ "High Latency" alerts per hour
- 50+ "Low Confidence" alerts per hour
- 30+ "Error Rate Spike" alerts per hour
- System appears to be functioning normally to users

**Analyze the alerting system:**
- Why this alert storm is occurring
- How to investigate whether alerts indicate real issues
- Strategies to reduce false positive alerts
- Improved alerting configuration recommendations

---

## Section D: Code Implementation (25 points, 5 points each)

**Answer Format: Complete, functional debugging and monitoring code**

26. **Training Diagnostics System (5 points)**
Complete the training diagnostics class that automatically detects common training issues:

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class TrainingDiagnostics:
    def __init__(self, patience=10):
        self.patience = patience
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=100)
        self.gradient_norms = deque(maxlen=100)

    def update(self, train_loss, val_loss, gradient_norm):
        """Update with latest training metrics"""
        ________________
        ________________
        ________________

    def diagnose_training_issues(self):
        """Diagnose common training problems"""
        if len(self.train_losses) < self.patience:
            return {"status": "insufficient_data"}

        issues = []

        # Check for stagnation
        recent_train = ________________
        train_trend = ________________  # Calculate slope

        if abs(train_trend) < 1e-6:
            issues.append({
                "issue": "training_stagnation",
                "description": "Training loss not decreasing",
                "recommendation": "Increase learning rate or check data quality"
            })

        # Check for overfitting
        recent_val = ________________
        val_trend = ________________

        if train_trend < -1e-4 and val_trend > 1e-4:
            issues.append({
                "issue": "overfitting",
                "description": "Validation loss increasing while training loss decreases",
                "recommendation": "Reduce model complexity or add regularization"
            })

        # Check for exploding gradients
        recent_grads = ________________
        if any(grad > 100 for grad in recent_grads):
            issues.append({
                "issue": "exploding_gradients",
                "description": "Gradient norms too high",
                "recommendation": "Use gradient clipping or reduce learning rate"
            })

        # Check for vanishing gradients
        if all(grad < 1e-7 for grad in recent_grads):
            issues.append({
                "issue": "vanishing_gradients",
                "description": "Gradient norms too small",
                "recommendation": "Check model initialization or use residual connections"
            })

        return {
            "status": "diagnosed",
            "issues": issues,
            "metrics": {
                "train_loss_trend": train_trend,
                "val_loss_trend": val_trend,
                "avg_gradient_norm": ________________
            }
        }

    def plot_diagnostics(self, save_path=None):
        """Plot diagnostic charts"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(list(self.train_losses), label='Train Loss')
        axes[0, 0].plot(list(self.val_losses), label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()

        # Gradient norms
        axes[0, 1].plot(list(self.gradient_norms))
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].set_yscale('log')

        # Loss difference (overfitting indicator)
        if len(self.train_losses) == len(self.val_losses):
            loss_diff = ________________
            axes[1, 0].plot(loss_diff)
            axes[1, 0].set_title('Val - Train Loss (Overfitting Indicator)')

        # Moving average of losses
        if len(self.train_losses) >= 10:
            train_ma = ________________  # Calculate 10-period moving average
            axes[1, 1].plot(train_ma, label='Train Loss MA')
            axes[1, 1].set_title('Moving Average Losses')
            axes[1, 1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
```

27. **Production Model Monitor (5 points)**
Complete the real-time production monitoring system:

```python
import time
from collections import deque, defaultdict
import numpy as np
from scipy import stats

class ProductionMonitor:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.feature_distributions = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_distributions = {}

    def set_baseline_distributions(self, baseline_data):
        """Set baseline feature distributions from training data"""
        ________________

    def record_prediction(self, features, prediction, confidence, latency):
        """Record a single prediction for monitoring"""
        ________________
        ________________
        ________________

        # Track feature distributions
        for feature_name, value in features.items():
            ________________

    def detect_data_drift(self, significance_level=0.05):
        """Detect data drift using statistical tests"""
        if not self.baseline_distributions:
            return {"status": "no_baseline", "drifts": []}

        drifts = []

        for feature_name, baseline_dist in self.baseline_distributions.items():
            current_dist = list(self.feature_distributions[feature_name])

            if len(current_dist) < 100:  # Need sufficient data
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = ________________

            if p_value < significance_level:
                drifts.append({
                    "feature": feature_name,
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "drift_magnitude": ________________  # How significant
                })

        return {
            "status": "analyzed",
            "drifts": drifts,
            "drift_detected": len(drifts) > 0
        }

    def check_performance_anomalies(self):
        """Check for performance anomalies"""
        if len(self.confidences) < 100:
            return {"status": "insufficient_data"}

        recent_confidences = list(self.confidences)[-100:]
        recent_latencies = list(self.latencies)[-100:]

        # Calculate statistics
        avg_confidence = ________________
        confidence_std = ________________
        avg_latency = ________________
        p95_latency = ________________

        anomalies = []

        # Check for low confidence
        if avg_confidence < 0.7:
            anomalies.append({
                "type": "low_confidence",
                "value": avg_confidence,
                "threshold": 0.7,
                "severity": "high" if avg_confidence < 0.5 else "medium"
            })

        # Check for high latency
        if p95_latency > 2.0:  # 2 second threshold
            anomalies.append({
                "type": "high_latency",
                "value": p95_latency,
                "threshold": 2.0,
                "severity": "high" if p95_latency > 5.0 else "medium"
            })

        return {
            "status": "analyzed",
            "anomalies": anomalies,
            "metrics": {
                "avg_confidence": avg_confidence,
                "confidence_std": confidence_std,
                "avg_latency": avg_latency,
                "p95_latency": p95_latency
            }
        }

    def generate_alert(self, alert_type, severity, message, value):
        """Generate structured alert"""
        return {
            "timestamp": time.time(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "value": value,
            "alert_id": f"{alert_type}_{int(time.time())}"
        }
```

28. **Attention Visualization for Debugging (5 points)**
Complete the attention visualization tool for transformer debugging:

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionDebugger:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_attention_weights(self, text, layer=-1, head=0):
        """Extract attention weights for analysis"""
        # Tokenize input
        inputs = ________________

        # Forward pass with attention output
        with torch.no_grad():
            outputs = ________________

        # Extract attention weights
        attention = ________________  # [seq_len, seq_len]
        tokens = ________________

        return attention.numpy(), tokens

    def analyze_attention_patterns(self, texts, layer=-1):
        """Analyze attention patterns across multiple examples"""
        attention_stats = {
            'entropy_scores': [],
            'self_attention_scores': [],
            'max_attention_positions': [],
            'attention_spread': []
        }

        for text in texts:
            attention, tokens = self.extract_attention_weights(text, layer)

            # Calculate attention entropy (how focused is attention)
            entropy = ________________  # -sum(attention * log(attention))
            attention_stats['entropy_scores'].append(entropy.mean())

            # Self-attention (diagonal values)
            self_attn = ________________
            attention_stats['self_attention_scores'].append(self_attn)

            # Position of maximum attention
            max_pos = ________________
            attention_stats['max_attention_positions'].extend(max_pos)

            # Attention spread (standard deviation)
            spread = ________________
            attention_stats['attention_spread'].append(spread.mean())

        return {
            'avg_entropy': np.mean(attention_stats['entropy_scores']),
            'avg_self_attention': np.mean(attention_stats['self_attention_scores']),
            'attention_position_distribution': np.histogram(attention_stats['max_attention_positions'], bins=10)[0],
            'avg_attention_spread': np.mean(attention_stats['attention_spread'])
        }

    def debug_misclassified_examples(self, texts, true_labels, predictions):
        """Debug attention patterns for misclassified examples"""
        misclassified_indices = ________________

        debug_results = []

        for idx in misclassified_indices:
            text = texts[idx]
            true_label = true_labels[idx]
            predicted_label = predictions[idx]

            attention, tokens = self.extract_attention_weights(text)

            # Find tokens with highest attention
            token_attention = ________________  # Sum attention for each token
            top_attended_tokens = ________________  # Get top 5

            debug_results.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'top_attended_tokens': top_attended_tokens,
                'attention_entropy': ________________,
                'max_attention_score': token_attention.max()
            })

        return debug_results

    def plot_attention_heatmap(self, text, layer=-1, head=0, save_path=None):
        """Plot attention heatmap for debugging"""
        attention, tokens = self.extract_attention_weights(text, layer, head)

        # Remove special tokens for cleaner visualization
        clean_tokens = []
        clean_attention = []

        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                clean_tokens.append(token.replace('##', ''))

        # Adjust attention matrix size accordingly
        ________________

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            clean_attention,
            xticklabels=clean_tokens,
            yticklabels=clean_tokens,
            cmap='Blues',
            cbar=True
        )
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

29. **Alert Management System (5 points)**
Complete the intelligent alert management system:

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Callable
import time
from collections import defaultdict

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

class IntelligentAlertManager:
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_counts = defaultdict(int)
        self.last_alert_times = defaultdict(float)
        self.suppression_rules = {}

    def create_alert(self, severity: AlertSeverity, title: str, description: str, source: str):
        """Create alert with intelligent suppression"""

        # Check suppression rules
        alert_key = f"{source}:{title}"
        current_time = time.time()

        # Suppress duplicate alerts within time window
        if alert_key in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_key]
            suppression_window = self._get_suppression_window(severity)

            if time_since_last < suppression_window:
                self.alert_counts[alert_key] += 1
                return None  # Suppressed

        # Create new alert
        alert_id = f"alert_{int(current_time)}_{len(self.alerts)}"
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=current_time,
            source=source
        )

        ________________
        ________________
        ________________

        # Check for alert storm and adjust severity
        if self.alert_counts[alert_key] > 10:
            ________________  # Escalate to higher severity

        return alert_id

    def _get_suppression_window(self, severity: AlertSeverity) -> float:
        """Get suppression window based on severity"""
        suppression_times = {
            AlertSeverity.LOW: ________________,      # 30 minutes
            AlertSeverity.MEDIUM: ________________,   # 15 minutes
            AlertSeverity.HIGH: ________________,     # 5 minutes
            AlertSeverity.CRITICAL: ________________  # 1 minute
        }
        return suppression_times.get(severity, 300)

    def analyze_alert_patterns(self) -> Dict:
        """Analyze alert patterns to identify systemic issues"""
        if not self.alerts:
            return {"status": "no_alerts"}

        # Group alerts by source and time
        alerts_by_source = defaultdict(list)
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]  # Last hour

        for alert in recent_alerts:
            ________________

        # Identify potential alert storms
        alert_storms = []
        for source, source_alerts in alerts_by_source.items():
            if len(source_alerts) > 20:  # More than 20 alerts per hour
                alert_storms.append({
                    "source": source,
                    "alert_count": len(source_alerts),
                    "severity_distribution": ________________,
                    "recommendation": "Investigate underlying system issue"
                })

        # Identify most common alert types
        alert_types = defaultdict(int)
        for alert in recent_alerts:
            ________________

        return {
            "status": "analyzed",
            "total_recent_alerts": len(recent_alerts),
            "alert_storms": alert_storms,
            "most_common_alerts": ________________,  # Top 5 most common
            "recommendations": self._generate_recommendations(alert_storms)
        }

    def _generate_recommendations(self, alert_storms: List[Dict]) -> List[str]:
        """Generate recommendations based on alert patterns"""
        recommendations = []

        if len(alert_storms) > 3:
            recommendations.append("Multiple alert storms detected - check system health")

        if any(storm["alert_count"] > 50 for storm in alert_storms):
            recommendations.append("Severe alert storm - consider emergency response")

        ________________  # Add more intelligent recommendations

        return recommendations

    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[Alert]:
        """Get active alerts with optional severity filter"""
        active_alerts = [a for a in self.alerts if not a.resolved]

        if severity_filter:
            active_alerts = ________________

        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
```

30. **Memory Profiling and Optimization (5 points)**
Complete the memory profiling system for training and inference:

```python
import psutil
import torch
import gc
from typing import Dict, List
import time

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process()

    def capture_snapshot(self, label: str = ""):
        """Capture memory snapshot"""
        # System memory
        system_memory = ________________
        process_memory = ________________

        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': ________________,
                'cached': ________________,
                'max_allocated': ________________
            }

        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'system_memory_gb': system_memory / (1024**3),
            'process_memory_gb': process_memory / (1024**3),
            'gpu_memory_gb': {k: v / (1024**3) for k, v in gpu_memory.items()}
        }

        self.snapshots.append(snapshot)
        return snapshot

    def analyze_memory_usage(self) -> Dict:
        """Analyze memory usage patterns"""
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data"}

        # Calculate memory trends
        process_memory_trend = []
        gpu_memory_trend = []

        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]

            process_change = ________________
            process_memory_trend.append(process_change)

            if curr['gpu_memory_gb']:
                gpu_change = ________________
                gpu_memory_trend.append(gpu_change)

        # Detect memory leaks
        memory_leak_detected = False
        if len(process_memory_trend) >= 5:
            # Check if memory consistently increases
            positive_trends = sum(1 for change in process_memory_trend[-5:] if change > 0.1)
            if positive_trends >= 4:
                memory_leak_detected = True

        # Find peak memory usage
        peak_process_memory = ________________
        peak_gpu_memory = ________________

        return {
            "status": "analyzed",
            "memory_leak_detected": memory_leak_detected,
            "peak_process_memory_gb": peak_process_memory,
            "peak_gpu_memory_gb": peak_gpu_memory,
            "avg_memory_change_per_snapshot": ________________,
            "total_snapshots": len(self.snapshots),
            "recommendations": self._generate_memory_recommendations()
        }

    def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []

        latest = self.snapshots[-1]

        # Check for high memory usage
        if latest['process_memory_gb'] > 8:
            recommendations.append("High process memory usage - consider reducing batch size")

        if latest['gpu_memory_gb'].get('allocated', 0) > 10:
            recommendations.append("High GPU memory usage - consider gradient checkpointing")

        # Check for memory growth
        if len(self.snapshots) >= 3:
            memory_growth = ________________
            if memory_growth > 1.0:  # More than 1GB growth
                recommendations.append("Memory usage growing - check for memory leaks")

        return recommendations

    def optimize_memory(self):
        """Perform memory optimization"""
        optimization_results = {}

        # Clear Python garbage collection
        collected = ________________
        optimization_results['gc_collected'] = collected

        # Clear PyTorch cache
        if torch.cuda.is_available():
            before_gpu = torch.cuda.memory_allocated()
            ________________
            after_gpu = torch.cuda.memory_allocated()
            optimization_results['gpu_memory_freed_gb'] = (before_gpu - after_gpu) / (1024**3)

        # Capture snapshot after optimization
        post_optimization_snapshot = self.capture_snapshot("post_optimization")
        optimization_results['snapshot'] = post_optimization_snapshot

        return optimization_results

    def monitor_training_memory(self, train_func, *args, **kwargs):
        """Monitor memory during training function"""
        self.capture_snapshot("training_start")

        try:
            result = train_func(*args, **kwargs)
            self.capture_snapshot("training_end")
            return result
        except Exception as e:
            self.capture_snapshot("training_error")
            analysis = self.analyze_memory_usage()

            if analysis.get("memory_leak_detected"):
                print("Memory leak detected during training!")

            raise e
```

---

## Scoring Rubric

### True/False (20 points)
- 2 points per correct answer

### Multiple Choice (25 points)
- 2.5 points per correct answer

### Debugging Scenarios (30 points)
- 6 points per scenario:
  - 6 points: Complete analysis with accurate diagnosis and practical solutions
  - 5 points: Good analysis with minor gaps in solution details
  - 4 points: Adequate diagnosis with basic solutions
  - 3 points: Partial understanding with some correct insights
  - 2 points: Minimal understanding demonstrated
  - 0-1 points: Incorrect analysis or missing

### Code Implementation (25 points)
- 5 points per question:
  - 5 points: Production-ready code that addresses the debugging/monitoring need effectively
  - 4 points: Mostly correct with minor implementation issues
  - 3 points: Shows understanding but has significant errors
  - 2 points: Partially correct with major issues
  - 1 point: Minimal understanding demonstrated
  - 0 points: Incorrect or missing

### Grade Scale
- A: 90-100 points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points

---

## Additional Guidance for Instructors

This assessment focuses on practical debugging and monitoring skills essential for production ML systems. Key areas evaluated include:

1. **Diagnostic Skills**: Ability to analyze symptoms and identify root causes
2. **Monitoring Design**: Understanding of what metrics to track and how
3. **Production Debugging**: Real-world problem-solving capabilities
4. **Code Quality**: Implementation of robust monitoring and debugging tools
5. **System Thinking**: Understanding interconnections between model performance and system health