# Assessment 5: Production Deployment
**Time Limit: 65 minutes**
**Total Points: 100**

## Instructions
- Answer all questions in the format specified
- For True/False: Write "TRUE" or "FALSE"
- For Multiple Choice: Write the letter (A, B, C, D, E)
- For Free Response: Provide detailed explanations with code examples
- For System Design: Draw/describe complete architectures

---

## Section A: True/False (20 points, 2 points each)

**Answer Format: TRUE or FALSE**

1. Batch inference is always more cost-effective than real-time inference for production systems.

2. Model quantization always results in significant accuracy degradation.

3. Kubernetes Horizontal Pod Autoscaler can automatically scale ML model pods based on CPU usage.

4. ONNX models typically have faster inference than PyTorch models in production.

5. A/B testing in ML requires the same users to always receive the same model version.

6. FastAPI automatically handles request validation and serialization for ML APIs.

7. Redis caching can improve ML API response times but increases memory usage.

8. Docker containers ensure identical behavior across development and production environments.

9. Model versioning is only necessary when model accuracy changes significantly.

10. Rate limiting is more important for ML APIs than traditional web APIs.

---

## Section B: Multiple Choice (25 points, 2.5 points each)

**Answer Format: Letter (A, B, C, D, or E)**

11. Which deployment strategy is best for a model serving 1M+ requests per day with strict latency requirements?
    A) Batch processing every hour
    B) Real-time API with horizontal scaling
    C) Single server with powerful GPU
    D) Serverless functions
    E) Database-stored predictions

12. What is the primary advantage of using ONNX for model deployment?
    A) Better accuracy than original models
    B) Cross-platform inference optimization
    C) Automatic model compression
    D) Built-in A/B testing
    E) Simplified training process

13. For model caching in production, which cache key strategy is most appropriate?
    A) Use only the input text
    B) Hash of input text + model version
    C) Use timestamp as key
    D) Use user ID as key
    E) Use random keys

14. In Kubernetes, what resource should you primarily monitor for ML model pods?
    A) CPU usage only
    B) Memory usage only
    C) Both CPU and memory usage
    D) Network bandwidth only
    E) Disk I/O only

15. Which authentication method is most suitable for high-frequency ML API calls?
    A) Username/password for each request
    B) JWT tokens with reasonable expiration
    C) OAuth2 for every request
    D) API keys with rate limiting
    E) No authentication needed

16. For A/B testing ML models, how should you assign users to model versions?
    A) Randomly for each request
    B) Based on user preferences
    C) Consistent assignment using user ID hash
    D) Time-based rotation
    E) Geographic location

17. What is the most critical metric to monitor for ML model APIs in production?
    A) Request count only
    B) Average response time only
    C) Error rate only
    D) Model accuracy in real-time
    E) Combination of latency, error rate, and throughput

18. When implementing graceful degradation for ML APIs, what should happen when the model fails?
    A) Return HTTP 500 error
    B) Return cached predictions
    C) Return default/fallback predictions
    D) Retry the request indefinitely
    E) Crash the service

19. For model versioning in production, which approach provides the best rollback capability?
    A) Overwrite the existing model file
    B) Use Git for model storage
    C) Immutable model artifacts with version tags
    D) Database storage of model weights
    E) Cloud storage with backup copies

20. What is the primary consideration when choosing between CPU and GPU for model inference?
    A) GPU is always better for any model
    B) Cost vs. latency requirements for your specific model
    C) CPU is sufficient for all transformer models
    D) GPU is only needed for training
    E) The choice doesn't affect performance

---

## Section C: System Design (30 points, 6 points each)

**Answer Format: Detailed architecture descriptions with justifications**

21. **High-Availability ML API Architecture (6 points)**
Design a production architecture for a text classification API that must handle:
- 10,000 requests per minute
- 99.9% uptime requirement
- <200ms response time (95th percentile)
- Global user base

**Required components to specify:**
- Load balancing strategy
- Scaling approach (horizontal/vertical)
- Caching layer design
- Failure handling mechanisms
- Geographic distribution considerations

22. **Model Deployment Pipeline (6 points)**
Design a CI/CD pipeline for ML models that automatically:
- Validates new model performance
- Deploys to staging environment
- Runs integration tests
- Promotes to production with zero downtime
- Provides rollback capability

**Required pipeline stages:**
- Model validation criteria
- Automated testing strategy
- Deployment methodology
- Monitoring integration
- Rollback triggers and process

23. **Multi-Model Serving System (6 points)**
Design a system that serves multiple ML models (sentiment analysis, intent classification, NER) with:
- Shared infrastructure
- Individual scaling per model
- Cost optimization
- Model version management

**Required architecture elements:**
- Resource sharing strategy
- Model routing mechanism
- Scaling policies per model
- Cost monitoring approach
- Version control system

24. **Edge Deployment Architecture (6 points)**
Design a deployment strategy for ML models that need to run on edge devices with:
- Limited computational resources (2GB RAM, ARM CPU)
- Intermittent internet connectivity
- Real-time inference requirements
- Periodic model updates

**Required specifications:**
- Model optimization techniques
- Local inference architecture
- Update mechanism design
- Offline capability handling
- Performance monitoring

25. **Security Architecture for ML APIs (6 points)**
Design a comprehensive security architecture for production ML APIs that handles:
- Authentication and authorization
- Input validation and sanitization
- Rate limiting and DDoS protection
- Data privacy and compliance
- Audit logging

**Required security components:**
- Authentication mechanism
- Input validation strategy
- Rate limiting implementation
- Data protection measures
- Logging and monitoring system

---

## Section D: Code Implementation (25 points, 5 points each)

**Answer Format: Complete, production-ready code**

26. **Production FastAPI Server (5 points)**
Complete the production-ready FastAPI server with proper error handling and monitoring:

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, validator
import time
import logging

app = FastAPI(title="ML Model API", version="1.0.0")

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str
    model_version: str = "latest"

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Text too long')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    processing_time_ms: float

# Global model instance
model_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_service
    ________________  # Initialize your model service

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded and responsive
        ________________
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    ________________

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    start_time = time.time()

    try:
        # Validate model is available
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model not available")

        # Make prediction
        result = ________________

        # Calculate processing time
        processing_time = ________________

        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            processing_time_ms=processing_time
        )

    except Exception as e:
        ________________  # Log error
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for monitoring"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    ________________  # Log request details

    return response
```

27. **Model Caching System (5 points)**
Complete the Redis-based caching system for ML predictions:

```python
import redis
import json
import hashlib
from typing import Optional, Dict

class ModelCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_client = ________________
        self.ttl = ttl

    def _generate_cache_key(self, text: str, model_version: str) -> str:
        """Generate consistent cache key"""
        content = ________________
        hash_key = ________________
        return f"prediction:{hash_key}"

    def get_prediction(self, text: str, model_version: str) -> Optional[Dict]:
        """Get cached prediction if available"""
        cache_key = ________________

        try:
            cached_result = ________________
            if cached_result:
                return ________________  # Parse JSON
            return None
        except Exception as e:
            ________________  # Log error but don't fail
            return None

    def cache_prediction(self, text: str, model_version: str, prediction: Dict):
        """Cache prediction result"""
        cache_key = ________________

        try:
            # Store with TTL
            ________________
        except Exception as e:
            ________________  # Log error but don't fail

    def invalidate_model_cache(self, model_version: str):
        """Invalidate all cache entries for a model version"""
        pattern = f"prediction:*:{model_version}:*"
        try:
            keys = ________________
            if keys:
                ________________  # Delete matching keys
        except Exception as e:
            ________________

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            info = ________________
            return {
                'total_keys': info.get('db0', {}).get('keys', 0),
                'memory_usage_mb': info.get('used_memory', 0) / 1024 / 1024,
                'hit_rate': ________________,  # Calculate from info
                'connected_clients': info.get('connected_clients', 0)
            }
        except Exception as e:
            return {'error': str(e)}
```

28. **A/B Testing Framework (5 points)**
Complete the A/B testing system for model deployment:

```python
import hashlib
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Experiment:
    name: str
    model_a: str
    model_b: str
    traffic_split: float
    start_date: datetime
    end_date: Optional[datetime] = None
    active: bool = True

class ABTestManager:
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.results: Dict[str, list] = {}

    def create_experiment(self, name: str, model_a: str, model_b: str,
                         traffic_split: float = 0.5) -> bool:
        """Create new A/B test experiment"""
        if name in self.experiments:
            return False

        experiment = Experiment(
            name=name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            start_date=datetime.now()
        )

        self.experiments[name] = experiment
        self.results[name] = []
        return True

    def get_model_version(self, experiment_name: str, user_id: str) -> str:
        """Determine which model version to use for this user"""
        if experiment_name not in self.experiments:
            return "default"

        experiment = self.experiments[experiment_name]
        if not experiment.active:
            return experiment.model_a  # Default to model A

        # Consistent assignment using user ID hash
        hash_value = ________________
        bucket = ________________  # Convert to 0-100 range

        if bucket < experiment.traffic_split * 100:
            return ________________
        else:
            return ________________

    def record_result(self, experiment_name: str, user_id: str, model_version: str,
                     prediction: int, confidence: float, latency: float):
        """Record experiment result"""
        if experiment_name not in self.experiments:
            return

        result = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'model_version': model_version,
            'prediction': prediction,
            'confidence': confidence,
            'latency': latency
        }

        ________________

    def get_experiment_results(self, experiment_name: str) -> Dict:
        """Get experiment results summary"""
        if experiment_name not in self.results:
            return {}

        results = self.results[experiment_name]
        if not results:
            return {}

        # Separate results by model
        model_a_results = ________________
        model_b_results = ________________

        return {
            'experiment': experiment_name,
            'total_requests': len(results),
            'model_a': {
                'requests': len(model_a_results),
                'avg_confidence': ________________,
                'avg_latency': ________________
            },
            'model_b': {
                'requests': len(model_b_results),
                'avg_confidence': ________________,
                'avg_latency': ________________
            }
        }

    def stop_experiment(self, experiment_name: str):
        """Stop running experiment"""
        if experiment_name in self.experiments:
            ________________
            ________________
```

29. **Kubernetes Deployment Configuration (5 points)**
Complete the Kubernetes deployment and service configuration:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  labels:
    app: ml-model-api
spec:
  replicas: ________________
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
    spec:
      containers:
      - name: ml-api
        image: ________________
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: ________________
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ________________
              key: ________________
        resources:
          requests:
            memory: ________________
            cpu: ________________
          limits:
            memory: ________________
            cpu: ________________
        livenessProbe:
          httpGet:
            path: ________________
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: ________________
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ________________
  ports:
  - protocol: TCP
    port: ________________
    targetPort: ________________
  type: ________________

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ________________
  minReplicas: ________________
  maxReplicas: ________________
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: ________________
```

30. **Production Monitoring Integration (5 points)**
Complete the monitoring system with Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import start_http_server
import time
from functools import wraps

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('ml_request_duration_seconds', 'Request latency')
PREDICTION_CONFIDENCE = Histogram('ml_prediction_confidence', 'Prediction confidence scores')
MODEL_LOAD_TIME = Gauge('ml_model_load_time_seconds', 'Model loading time')
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Active connections')
ERROR_RATE = Gauge('ml_error_rate', 'Error rate percentage')

class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.error_count = 0
        self.confidence_scores = []

    def record_request(self, endpoint: str, status: str, latency: float,
                      confidence: float = None):
        """Record request metrics"""
        # Update counters
        ________________
        ________________

        self.total_requests += 1
        if status == 'error':
            self.error_count += 1

        # Update confidence if provided
        if confidence is not None:
            ________________
            self.confidence_scores.append(confidence)

        # Update error rate
        error_rate = ________________
        ________________

    def metrics_middleware(self, func):
        """Decorator to automatically collect metrics"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            ________________  # Increment active connections

            try:
                result = await func(*args, **kwargs)
                status = 'success'
                confidence = getattr(result, 'confidence', None)
            except Exception as e:
                status = 'error'
                confidence = None
                raise
            finally:
                latency = ________________
                ________________  # Decrement active connections

                self.record_request(
                    endpoint=func.__name__,
                    status=status,
                    latency=latency,
                    confidence=confidence
                )

            return result

        return wrapper

    def get_metrics_summary(self) -> dict:
        """Get current metrics summary"""
        return {
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_requests, 1),
            'avg_confidence': ________________,
            'active_connections': ________________
        }

# Start Prometheus metrics server
def start_metrics_server(port: int = 8001):
    ________________

# Usage with FastAPI
metrics = MetricsCollector()

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(________________, media_type="text/plain")
```

---

## Scoring Rubric

### True/False (20 points)
- 2 points per correct answer

### Multiple Choice (25 points)
- 2.5 points per correct answer

### System Design (30 points)
- 6 points per question:
  - 6 points: Complete, detailed architecture with clear justifications
  - 5 points: Good design with minor gaps
  - 4 points: Adequate design showing understanding
  - 3 points: Basic design with some technical issues
  - 2 points: Minimal understanding shown
  - 0-1 points: Incorrect or missing

### Code Implementation (25 points)
- 5 points per question:
  - 5 points: Production-ready code that would work correctly
  - 4 points: Mostly correct with minor issues
  - 3 points: Shows understanding but has implementation errors
  - 2 points: Partially correct with significant issues
  - 1 point: Minimal understanding demonstrated
  - 0 points: Incorrect or missing

### Grade Scale
- A: 90-100 points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points