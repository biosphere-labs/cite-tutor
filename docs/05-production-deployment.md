# Production Deployment Considerations

## Model Serving Architecture

### Choosing Deployment Strategy

#### Batch vs Real-time Inference
```python
# Batch Processing (Good for: Analytics, ETL, Non-urgent predictions)
class BatchInferenceService:
    def __init__(self, model_path: str, batch_size: int = 1000):
        self.model = self.load_model(model_path)
        self.batch_size = batch_size

    def process_batch_file(self, input_file: str, output_file: str):
        """Process large files in batches"""
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            batch = []
            for line in f_in:
                batch.append(json.loads(line))

                if len(batch) >= self.batch_size:
                    predictions = self.predict_batch(batch)
                    for item, pred in zip(batch, predictions):
                        result = {**item, 'prediction': pred}
                        f_out.write(json.dumps(result) + '\n')
                    batch = []

            # Process remaining items
            if batch:
                predictions = self.predict_batch(batch)
                for item, pred in zip(batch, predictions):
                    result = {**item, 'prediction': pred}
                    f_out.write(json.dumps(result) + '\n')

# Real-time API (Good for: User-facing apps, Low-latency requirements)
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

class RealTimeInferenceService:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.model.eval()

    @app.route('/predict', methods=['POST'])
    def predict(self):
        try:
            data = request.json
            text = data.get('text', '')

            if not text:
                return jsonify({'error': 'No text provided'}), 400

            # Preprocess and predict
            prediction = self.predict_single(text)

            return jsonify({
                'prediction': prediction,
                'confidence': float(prediction.max()),
                'processing_time': time.time() - start_time
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
```

### Model Optimization for Production

#### Model Serialization and Loading
```python
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ProductionModel:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer, self.config = self.load_optimized_model(model_path)

    def load_optimized_model(self, model_path: str):
        """Load model optimized for production"""

        # Method 1: Load from checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint['model_name'],
            num_labels=checkpoint['num_labels']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['model_name'])

        # Enable optimizations
        if torch.cuda.is_available():
            model = model.half()  # Use FP16 for faster inference

        return model, tokenizer, checkpoint.get('config', {})

    def predict_optimized(self, texts: List[str]) -> List[Dict]:
        """Optimized prediction with batching"""

        # Tokenize in batch
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Inference with no gradient computation
        with torch.no_grad():
            outputs = self.model(**encoding)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        # Convert to Python types
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'prediction': int(predictions[i]),
                'confidence': float(probabilities[i].max()),
                'all_scores': probabilities[i].cpu().numpy().tolist()
            })

        return results
```

#### TensorRT and ONNX Optimization
```python
import onnx
import onnxruntime as ort
import torch

class ONNXOptimizedModel:
    def __init__(self, pytorch_model_path: str, onnx_model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if onnx_model_path and os.path.exists(onnx_model_path):
            self.load_onnx_model(onnx_model_path)
        else:
            self.convert_to_onnx(pytorch_model_path, onnx_model_path or 'model.onnx')

    def convert_to_onnx(self, pytorch_model_path: str, onnx_output_path: str):
        """Convert PyTorch model to ONNX for optimization"""

        # Load PyTorch model
        model = torch.load(pytorch_model_path, map_location=self.device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, 512)).to(self.device)  # Batch size 1, seq len 512

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
        )

        self.load_onnx_model(onnx_output_path)

    def load_onnx_model(self, onnx_model_path: str):
        """Load ONNX model for inference"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

    def predict_onnx(self, input_ids: np.ndarray) -> np.ndarray:
        """Fast inference using ONNX runtime"""
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_ids}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs[0]
```

## Containerization and Orchestration

### Docker Setup
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "src/app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/best_model.pt
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  redis_data:
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  labels:
    app: ml-model-api
spec:
  replicas: 3
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
        image: your-registry/ml-model-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.pt"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## API Design and Implementation

### FastAPI Production Server
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, validator
import asyncio
import time
from typing import List, Optional
import uvicorn

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

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    model_version: Optional[str] = "latest"

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    processing_time_ms: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model_version: Optional[str] = "latest"

    @validator('texts')
    def texts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        if len(v) > 100:  # Limit batch size
            raise ValueError('Batch size cannot exceed 100')
        return v

# Global model instance
model_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_service
    model_service = ProductionModel(MODEL_PATH)
    print("Model loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    start_time = time.time()

    try:
        result = model_service.predict_optimized([request.text])[0]
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            processing_time_ms=processing_time,
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    start_time = time.time()

    try:
        results = model_service.predict_optimized(request.texts)
        processing_time = (time.time() - start_time) * 1000

        return {
            "predictions": results,
            "processing_time_ms": processing_time,
            "model_version": "1.0.0",
            "batch_size": len(request.texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for monitoring"""
    # This would typically integrate with Prometheus
    return {
        "requests_total": 1000,  # Example metrics
        "requests_per_second": 10.5,
        "average_latency_ms": 150,
        "error_rate": 0.01
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Number of worker processes
        loop="uvloop",  # High-performance event loop
        log_level="info"
    )
```

### Caching and Rate Limiting
```python
import redis
from functools import wraps
import hashlib
import json

class CacheService:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour

    def get_cache_key(self, text: str, model_version: str) -> str:
        """Generate cache key for prediction"""
        content = f"{text}:{model_version}"
        return f"prediction:{hashlib.md5(content.encode()).hexdigest()}"

    def get_cached_prediction(self, text: str, model_version: str) -> Optional[Dict]:
        """Get cached prediction if available"""
        cache_key = self.get_cache_key(text, model_version)
        cached = self.redis_client.get(cache_key)

        if cached:
            return json.loads(cached)
        return None

    def cache_prediction(self, text: str, model_version: str, prediction: Dict):
        """Cache prediction result"""
        cache_key = self.get_cache_key(text, model_version)
        self.redis_client.setex(
            cache_key,
            self.default_ttl,
            json.dumps(prediction)
        )

# Rate limiting decorator
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check if under limit
        if len(self.requests[client_id]) < self.requests_per_minute:
            self.requests[client_id].append(now)
            return True

        return False

# Integration with FastAPI
cache_service = CacheService()
rate_limiter = RateLimiter(requests_per_minute=100)

@app.post("/predict_cached", response_model=PredictionResponse)
async def predict_with_cache(request: PredictionRequest, client_ip: str = Depends(get_client_ip)):
    """Prediction with caching and rate limiting"""

    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Check cache
    cached_result = cache_service.get_cached_prediction(request.text, request.model_version)
    if cached_result:
        return PredictionResponse(**cached_result)

    # Make prediction
    start_time = time.time()
    result = model_service.predict_optimized([request.text])[0]
    processing_time = (time.time() - start_time) * 1000

    response = PredictionResponse(
        prediction=result['prediction'],
        confidence=result['confidence'],
        processing_time_ms=processing_time,
        model_version="1.0.0"
    )

    # Cache result
    cache_service.cache_prediction(request.text, request.model_version, response.dict())

    return response
```

## Monitoring and Observability

### Application Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('ml_request_duration_seconds', 'Request latency')
PREDICTION_CONFIDENCE = Histogram('ml_prediction_confidence', 'Prediction confidence scores')
MODEL_LOAD_TIME = Gauge('ml_model_load_time_seconds', 'Model loading time')
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Active connections')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()

            # Increment request counter
            REQUEST_COUNT.labels(
                endpoint=scope["path"],
                method=scope["method"]
            ).inc()

            # Track active connections
            ACTIVE_CONNECTIONS.inc()

            try:
                await self.app(scope, receive, send)
            finally:
                # Record latency
                REQUEST_LATENCY.observe(time.time() - start_time)
                ACTIVE_CONNECTIONS.dec()
        else:
            await self.app(scope, receive, send)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

### Logging and Error Tracking
```python
import logging
import structlog
from pythonjsonlogger import jsonlogger

# Configure structured logging
def setup_logging():
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()

# Error tracking
class ErrorTracker:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_examples = defaultdict(list)

    def track_error(self, error_type: str, error_msg: str, context: Dict = None):
        self.error_counts[error_type] += 1

        example = {
            'message': error_msg,
            'timestamp': time.time(),
            'context': context or {}
        }

        # Keep only last 10 examples per error type
        self.error_examples[error_type].append(example)
        if len(self.error_examples[error_type]) > 10:
            self.error_examples[error_type].pop(0)

        logger.error(
            "prediction_error",
            error_type=error_type,
            error_message=error_msg,
            context=context
        )

error_tracker = ErrorTracker()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    logger.info(
        "request_started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host
    )

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        "request_completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )

    return response
```

## Model Versioning and A/B Testing

### Model Registry
```python
class ModelRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.models = {}
        self.metadata = {}

    def register_model(self, model_name: str, version: str, model_path: str, metadata: Dict):
        """Register a new model version"""
        key = f"{model_name}:{version}"

        self.models[key] = {
            'path': model_path,
            'loaded_at': time.time(),
            'model': None  # Lazy loading
        }

        self.metadata[key] = {
            'name': model_name,
            'version': version,
            'registered_at': time.time(),
            'metrics': metadata.get('metrics', {}),
            'training_data': metadata.get('training_data', {}),
            'config': metadata.get('config', {})
        }

    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load model (with caching)"""
        if version == "latest":
            version = self.get_latest_version(model_name)

        key = f"{model_name}:{version}"

        if key not in self.models:
            raise ValueError(f"Model {key} not found")

        if self.models[key]['model'] is None:
            self.models[key]['model'] = self._load_model_from_path(
                self.models[key]['path']
            )

        return self.models[key]['model']

    def get_latest_version(self, model_name: str) -> str:
        """Get latest version of a model"""
        versions = [
            meta['version'] for meta in self.metadata.values()
            if meta['name'] == model_name
        ]

        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        # Sort versions (assuming semantic versioning)
        return sorted(versions, key=lambda x: tuple(map(int, x.split('.'))))[-1]

# A/B Testing Framework
class ABTestManager:
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.experiments = {}

    def create_experiment(self, experiment_name: str, model_a: str, model_b: str, traffic_split: float = 0.5):
        """Create A/B test experiment"""
        self.experiments[experiment_name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': [],
            'created_at': time.time()
        }

    def get_model_for_request(self, experiment_name: str, user_id: str) -> str:
        """Determine which model to use for this request"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]

        # Use consistent hashing to ensure same user always gets same model
        hash_value = hash(user_id) % 100

        if hash_value < experiment['traffic_split'] * 100:
            return experiment['model_a']
        else:
            return experiment['model_b']

    def record_result(self, experiment_name: str, model_version: str, result: Dict):
        """Record experiment result"""
        if experiment_name not in self.experiments:
            return

        experiment = self.experiments[experiment_name]

        if model_version == experiment['model_a']:
            experiment['results_a'].append(result)
        elif model_version == experiment['model_b']:
            experiment['results_b'].append(result)

# Integration with API
model_registry = ModelRegistry("/app/models")
ab_test_manager = ABTestManager(model_registry)

@app.post("/predict_ab")
async def predict_with_ab_test(
    request: PredictionRequest,
    user_id: str = Header(...),
    experiment: str = Query("default")
):
    """Prediction with A/B testing"""

    # Determine model version
    model_version = ab_test_manager.get_model_for_request(experiment, user_id)

    # Load appropriate model
    model = model_registry.load_model("sentiment_classifier", model_version)

    # Make prediction
    start_time = time.time()
    result = model.predict([request.text])[0]
    processing_time = time.time() - start_time

    # Record result for A/B test
    ab_test_manager.record_result(experiment, model_version, {
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'processing_time': processing_time,
        'timestamp': time.time()
    })

    return PredictionResponse(
        prediction=result['prediction'],
        confidence=result['confidence'],
        processing_time_ms=processing_time * 1000,
        model_version=model_version
    )
```

## Security Considerations

### Authentication and Authorization
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def verify_token(self, token: str) -> Dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def check_permissions(self, user_payload: Dict, required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = user_payload.get('permissions', [])
        return required_permission in user_permissions

auth_manager = AuthManager(SECRET_KEY)

def require_auth(required_permission: str = "predict"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, credentials: HTTPAuthorizationCredentials = Depends(security), **kwargs):
            user_payload = auth_manager.verify_token(credentials.credentials)

            if not auth_manager.check_permissions(user_payload, required_permission):
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/predict_secure")
@require_auth("predict")
async def secure_predict(request: PredictionRequest):
    """Secure prediction endpoint"""
    # Implementation here
    pass
```

### Input Validation and Sanitization
```python
import re
from typing import List

class InputValidator:
    def __init__(self):
        self.max_text_length = 10000
        self.allowed_chars_pattern = re.compile(r'^[\w\s\.,!?;:\-()"\'\n]+$', re.UNICODE)

    def validate_text(self, text: str) -> str:
        """Validate and sanitize input text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > self.max_text_length:
            raise ValueError(f"Text too long. Maximum {self.max_text_length} characters allowed")

        # Basic sanitization
        text = text.strip()

        # Remove potentially malicious patterns
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)

        return text

    def validate_batch(self, texts: List[str]) -> List[str]:
        """Validate batch of texts"""
        if len(texts) > 100:
            raise ValueError("Batch size too large. Maximum 100 texts allowed")

        return [self.validate_text(text) for text in texts]

validator = InputValidator()

# Use in endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    validated_text = validator.validate_text(request.text)
    # Continue with prediction...
```

## Next Steps

Continue to [Debugging and Monitoring](06-debugging-monitoring.md) to learn advanced techniques for troubleshooting and maintaining AI systems in production.