# Model Training Configuration for Cite-Tutor

## Overview

This document provides comprehensive guidance on using the `config/models.yaml` file to configure and train AI models for the Cite-Tutor system. The configuration is optimized for 4GB VRAM GPUs while maintaining training quality and supporting various model components.

## Configuration Structure

### models.yaml Configuration

The `config/models.yaml` file defines model configurations optimized for resource-constrained environments:

```yaml
# 4GB VRAM optimized models
structure_analysis:
  model: "google/flan-t5-small"  # 60MB
  max_memory_mb: 1024

qa_generation:
  model: "google/flan-t5-small"
  batch_size: 1
  max_length: 128

fine_tuning:
  base_model: "distilgpt2"  # 82MB
  quantization: "4bit"
  lora_r: 16
  lora_alpha: 32
  batch_size: 1
  gradient_accumulation: 32

embedding:
  model: "all-MiniLM-L6-v2"  # 22MB, for RAG

memory_limits:
  max_gpu_memory_mb: 4096
  safety_buffer_mb: 512
```

## Model Components

### 1. Structure Analysis Model

**Purpose**: Analyzes document structure and extracts sections from academic papers.

**Configuration**:
- **Model**: `google/flan-t5-small` (60MB)
- **Memory Limit**: 1024MB
- **Use Case**: Document parsing, section identification

**Training on AWS Spot Fleet**:
```python
# AWS training job for structure analysis
structure_job = {
    "job_id": "sci-tutor-structure-001",
    "model_type": "structure_analysis",
    "model_config": {
        "model": "google/flan-t5-small",
        "max_memory_mb": 1024,
        "batch_size": 2,  # Can increase on GPU instances
        "learning_rate": 1e-4,
        "epochs": 3
    },
    "data_s3_path": "s3://sci-tutor-data/structure/",
    "instance_type": "g4dn.xlarge",
    "max_runtime_minutes": 30
}
```

### 2. Q&A Generation Model

**Purpose**: Generates question-answer pairs from processed academic content.

**Configuration**:
- **Model**: `google/flan-t5-small`
- **Batch Size**: 1 (local), 2-4 (AWS)
- **Max Length**: 128 tokens
- **Use Case**: Creating training data, content understanding

**Training on AWS Spot Fleet**:
```python
# AWS training job for Q&A generation
qa_job = {
    "job_id": "sci-tutor-qa-001",
    "model_type": "qa_generation",
    "model_config": {
        "model": "google/flan-t5-small",
        "batch_size": 4,  # Increased for GPU
        "max_length": 256,  # Can increase on GPU
        "learning_rate": 5e-5,
        "epochs": 2
    },
    "data_s3_path": "s3://sci-tutor-data/qa-pairs/",
    "instance_type": "g4dn.xlarge",
    "max_runtime_minutes": 45
}
```

### 3. Fine-Tuning Model (Primary Training)

**Purpose**: Main conversational AI model for tutoring and domain expertise.

**Configuration**:
- **Base Model**: `distilgpt2` (82MB)
- **Quantization**: 4-bit for memory efficiency
- **LoRA**: Rank 16, Alpha 32 for parameter-efficient training
- **Gradient Accumulation**: 32 steps for effective larger batch size

**Multi-Stage Training Pipeline**:

#### Stage 1: Book Foundation Training
```python
stage1_job = {
    "job_id": "sci-tutor-stage1-001",
    "model_type": "fine_tuning",
    "stage": "book_foundation",
    "model_config": {
        "base_model": "distilgpt2",
        "quantization": "4bit",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,  # Increased for GPU
        "gradient_accumulation": 16,  # Reduced due to larger batch
        "learning_rate": 1e-4,
        "epochs": 3,
        "warmup_ratio": 0.1
    },
    "data_s3_path": "s3://sci-tutor-data/book-qa/",
    "output_s3_path": "s3://sci-tutor-models/stage1/",
    "instance_type": "g4dn.xlarge",
    "max_runtime_minutes": 120
}
```

#### Stage 2: Paper Integration Training
```python
stage2_job = {
    "job_id": "sci-tutor-stage2-001",
    "model_type": "fine_tuning",
    "stage": "paper_integration",
    "model_config": {
        "base_model_s3": "s3://sci-tutor-models/stage1/",
        "learning_rate": 5e-5,  # Lower to preserve knowledge
        "epochs": 2,
        "warmup_ratio": 0.05
    },
    "data_s3_path": "s3://sci-tutor-data/paper-qa/",
    "output_s3_path": "s3://sci-tutor-models/stage2/",
    "instance_type": "g4dn.xlarge",
    "max_runtime_minutes": 90
}
```

#### Stage 3: Knowledge Synthesis
```python
stage3_job = {
    "job_id": "sci-tutor-stage3-001",
    "model_type": "fine_tuning",
    "stage": "knowledge_synthesis",
    "model_config": {
        "base_model_s3": "s3://sci-tutor-models/stage2/",
        "learning_rate": 2e-5,  # Lowest for refinement
        "epochs": 1,
        "warmup_ratio": 0.03
    },
    "data_s3_path": "s3://sci-tutor-data/integrated-qa/",
    "output_s3_path": "s3://sci-tutor-models/final/",
    "instance_type": "g4dn.xlarge",
    "max_runtime_minutes": 60
}
```

### 4. Embedding Model

**Purpose**: Creates vector embeddings for RAG (Retrieval-Augmented Generation) system.

**Configuration**:
- **Model**: `all-MiniLM-L6-v2` (22MB)
- **Use Case**: Document search, citation lookup, content retrieval

**Training on AWS Spot Fleet**:
```python
embedding_job = {
    "job_id": "sci-tutor-embedding-001",
    "model_type": "embedding",
    "model_config": {
        "model": "all-MiniLM-L6-v2",
        "batch_size": 32,  # Large batch for embeddings
        "max_length": 512,
        "learning_rate": 2e-5,
        "epochs": 1
    },
    "data_s3_path": "s3://sci-tutor-data/documents/",
    "instance_type": "g4dn.xlarge",
    "max_runtime_minutes": 30
}
```

## AWS Spot Fleet Training Implementation

### 1. Orchestrator Script

Create `src/aws_orchestrator.py` to manage the training pipeline:

```python
#!/usr/bin/env python3
"""
AWS Spot Fleet Training Orchestrator for Sci-Tutor

This script coordinates model training across spot fleet instances,
managing job distribution, monitoring, and result aggregation.
"""

import boto3
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingJob:
    job_id: str
    model_type: str
    stage: Optional[str]
    model_config: Dict
    data_s3_path: str
    output_s3_path: str
    instance_type: str
    max_runtime_minutes: int
    priority: int = 1

class SciTutorOrchestrator:
    def __init__(self, config_path: str = "config/models.yaml"):
        self.config = self._load_config(config_path)
        self.sqs = boto3.client('sqs')
        self.ec2 = boto3.client('ec2')
        self.s3 = boto3.client('s3')

        # Queue URLs (set via environment or configuration)
        self.training_queue_url = os.getenv('TRAINING_QUEUE_URL')
        self.result_queue_url = os.getenv('RESULT_QUEUE_URL')

        # Spot fleet configuration
        self.spot_fleet_id = None
        self.target_capacity = 2
        self.max_spot_price = 0.50

    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_training_pipeline(self, domain: str = "chemistry") -> List[TrainingJob]:
        """Create complete training pipeline jobs."""
        jobs = []

        # Structure analysis job
        jobs.append(TrainingJob(
            job_id=f"sci-tutor-structure-{domain}-001",
            model_type="structure_analysis",
            stage=None,
            model_config=self.config['structure_analysis'],
            data_s3_path=f"s3://sci-tutor-data/{domain}/structure/",
            output_s3_path=f"s3://sci-tutor-models/{domain}/structure/",
            instance_type="g4dn.xlarge",
            max_runtime_minutes=30,
            priority=1
        ))

        # Q&A generation job
        jobs.append(TrainingJob(
            job_id=f"sci-tutor-qa-{domain}-001",
            model_type="qa_generation",
            stage=None,
            model_config=self.config['qa_generation'],
            data_s3_path=f"s3://sci-tutor-data/{domain}/qa-pairs/",
            output_s3_path=f"s3://sci-tutor-models/{domain}/qa/",
            instance_type="g4dn.xlarge",
            max_runtime_minutes=45,
            priority=2
        ))

        # Multi-stage fine-tuning jobs
        base_config = self.config['fine_tuning'].copy()

        # Stage 1: Book foundation
        jobs.append(TrainingJob(
            job_id=f"sci-tutor-stage1-{domain}-001",
            model_type="fine_tuning",
            stage="book_foundation",
            model_config={**base_config, "epochs": 3, "learning_rate": 1e-4},
            data_s3_path=f"s3://sci-tutor-data/{domain}/book-qa/",
            output_s3_path=f"s3://sci-tutor-models/{domain}/stage1/",
            instance_type="g4dn.xlarge",
            max_runtime_minutes=120,
            priority=3
        ))

        # Stage 2: Paper integration (depends on stage 1)
        jobs.append(TrainingJob(
            job_id=f"sci-tutor-stage2-{domain}-001",
            model_type="fine_tuning",
            stage="paper_integration",
            model_config={**base_config, "epochs": 2, "learning_rate": 5e-5,
                         "base_model_s3": f"s3://sci-tutor-models/{domain}/stage1/"},
            data_s3_path=f"s3://sci-tutor-data/{domain}/paper-qa/",
            output_s3_path=f"s3://sci-tutor-models/{domain}/stage2/",
            instance_type="g4dn.xlarge",
            max_runtime_minutes=90,
            priority=4
        ))

        # Stage 3: Knowledge synthesis (depends on stage 2)
        jobs.append(TrainingJob(
            job_id=f"sci-tutor-stage3-{domain}-001",
            model_type="fine_tuning",
            stage="knowledge_synthesis",
            model_config={**base_config, "epochs": 1, "learning_rate": 2e-5,
                         "base_model_s3": f"s3://sci-tutor-models/{domain}/stage2/"},
            data_s3_path=f"s3://sci-tutor-data/{domain}/integrated-qa/",
            output_s3_path=f"s3://sci-tutor-models/{domain}/final/",
            instance_type="g4dn.xlarge",
            max_runtime_minutes=60,
            priority=5
        ))

        # Embedding model job
        jobs.append(TrainingJob(
            job_id=f"sci-tutor-embedding-{domain}-001",
            model_type="embedding",
            stage=None,
            model_config=self.config['embedding'],
            data_s3_path=f"s3://sci-tutor-data/{domain}/documents/",
            output_s3_path=f"s3://sci-tutor-models/{domain}/embedding/",
            instance_type="g4dn.xlarge",
            max_runtime_minutes=30,
            priority=6
        ))

        return jobs

    def submit_jobs_to_queue(self, jobs: List[TrainingJob]):
        """Submit training jobs to SQS queue."""
        for job in jobs:
            message = {
                "job_id": job.job_id,
                "model_type": job.model_type,
                "stage": job.stage,
                "model_config": job.model_config,
                "data_s3_path": job.data_s3_path,
                "output_s3_path": job.output_s3_path,
                "instance_type": job.instance_type,
                "max_runtime_minutes": job.max_runtime_minutes,
                "priority": job.priority,
                "timestamp": time.time()
            }

            self.sqs.send_message(
                QueueUrl=self.training_queue_url,
                MessageBody=json.dumps(message),
                MessageAttributes={
                    'Priority': {
                        'StringValue': str(job.priority),
                        'DataType': 'Number'
                    },
                    'ModelType': {
                        'StringValue': job.model_type,
                        'DataType': 'String'
                    }
                }
            )

            logger.info(f"Submitted job: {job.job_id}")

    def create_spot_fleet(self):
        """Create spot fleet for training workers."""

        # Load user data script
        with open('user_data_script.sh', 'r') as f:
            user_data = f.read()

        # Replace placeholders
        user_data = user_data.replace('${sqs_queue_url}', self.training_queue_url)

        import base64
        user_data_encoded = base64.b64encode(user_data.encode()).decode()

        spot_fleet_config = {
            "SpotFleetRequestConfig": {
                "IamFleetRole": f"arn:aws:iam::{self._get_account_id()}:role/aws-ec2-spot-fleet-tagging-role",
                "AllocationStrategy": "diversified",
                "TargetCapacity": self.target_capacity,
                "SpotPrice": str(self.max_spot_price),
                "LaunchSpecifications": [
                    {
                        "ImageId": "ami-0c02fb55956c7d316",  # Deep Learning AMI
                        "InstanceType": "g4dn.xlarge",
                        "KeyName": "sci-tutor-key",
                        "SecurityGroups": [{"GroupId": "sg-xxxxxxxxx"}],
                        "IamInstanceProfile": {
                            "Arn": f"arn:aws:iam::{self._get_account_id()}:instance-profile/sci-tutor-worker-role"
                        },
                        "UserData": user_data_encoded,
                        "TagSpecifications": [{
                            "ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "sci-tutor-worker"}]
                        }]
                    }
                ],
                "TerminateInstancesWithExpiration": True,
                "Type": "maintain"
            }
        }

        response = self.ec2.request_spot_fleet(**spot_fleet_config)
        self.spot_fleet_id = response['SpotFleetRequestId']
        logger.info(f"Created spot fleet: {self.spot_fleet_id}")

    def monitor_training_progress(self):
        """Monitor training progress and manage fleet scaling."""
        while True:
            # Check queue depth
            queue_attrs = self.sqs.get_queue_attributes(
                QueueUrl=self.training_queue_url,
                AttributeNames=['ApproximateNumberOfMessages']
            )

            queue_depth = int(queue_attrs['Attributes']['ApproximateNumberOfMessages'])
            logger.info(f"Queue depth: {queue_depth}")

            # Auto-scale fleet based on queue depth
            if queue_depth > 5 and self.target_capacity < 4:
                self._scale_fleet(self.target_capacity + 1)
            elif queue_depth == 0 and self.target_capacity > 1:
                self._scale_fleet(self.target_capacity - 1)

            # Check for completed jobs
            self._process_completed_jobs()

            time.sleep(60)  # Check every minute

    def _scale_fleet(self, new_capacity: int):
        """Scale spot fleet to new capacity."""
        self.ec2.modify_spot_fleet_request(
            SpotFleetRequestId=self.spot_fleet_id,
            TargetCapacity=new_capacity
        )
        self.target_capacity = new_capacity
        logger.info(f"Scaled fleet to {new_capacity} instances")

    def _process_completed_jobs(self):
        """Process completed job notifications."""
        while True:
            response = self.sqs.receive_message(
                QueueUrl=self.result_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=1
            )

            messages = response.get('Messages', [])
            if not messages:
                break

            for message in messages:
                result = json.loads(message['Body'])
                logger.info(f"Job completed: {result['job_id']}, Status: {result['status']}")

                # Delete processed message
                self.sqs.delete_message(
                    QueueUrl=self.result_queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )

    def _get_account_id(self) -> str:
        """Get AWS account ID."""
        return boto3.client('sts').get_caller_identity()['Account']

def main():
    """Main orchestrator loop."""
    orchestrator = SciTutorOrchestrator()

    # Create training pipeline for chemistry domain
    jobs = orchestrator.create_training_pipeline("chemistry")

    # Submit jobs to queue
    orchestrator.submit_jobs_to_queue(jobs)

    # Create spot fleet
    orchestrator.create_spot_fleet()

    # Monitor progress
    orchestrator.monitor_training_progress()

if __name__ == "__main__":
    main()
```

### 2. Worker Training Script

Create enhanced worker script that handles different model types:

```python
# Enhanced worker script for user_data_script.sh
def process_ml_job(self, job_data):
    """Process ML job based on model type and configuration."""
    import json
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        T5Tokenizer, T5ForConditionalGeneration,
        Trainer, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    job = json.loads(job_data)
    model_type = job['model_type']
    model_config = job['model_config']

    logger.info(f"Processing {model_type} job: {job['job_id']}")

    try:
        if model_type == "structure_analysis":
            self._train_structure_model(job)
        elif model_type == "qa_generation":
            self._train_qa_model(job)
        elif model_type == "fine_tuning":
            self._train_fine_tuning_model(job)
        elif model_type == "embedding":
            self._train_embedding_model(job)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Upload results to S3
        self._upload_results(job)

        # Send completion notification
        self._notify_completion(job, "success")

    except Exception as e:
        logger.error(f"Job failed: {e}")
        self._notify_completion(job, "failed", str(e))

def _train_fine_tuning_model(self, job):
    """Train the main fine-tuning model with multi-stage support."""
    model_config = job['model_config']
    stage = job.get('stage', 'default')

    # Load base model or previous stage
    if 'base_model_s3' in model_config:
        # Download from S3
        self._download_from_s3(model_config['base_model_s3'], '/tmp/base_model')
        model = AutoModelForCausalLM.from_pretrained('/tmp/base_model')
        tokenizer = AutoTokenizer.from_pretrained('/tmp/base_model')
    else:
        model_name = model_config['base_model']
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply quantization if specified
    if model_config.get('quantization') == '4bit':
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config
        )

    # Apply LoRA
    lora_config = LoraConfig(
        r=model_config.get('lora_r', 16),
        lora_alpha=model_config.get('lora_alpha', 32),
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load and prepare training data
    dataset = self._load_training_data(job['data_s3_path'])

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=f'/tmp/training_{stage}',
        per_device_train_batch_size=model_config.get('batch_size', 2),
        gradient_accumulation_steps=model_config.get('gradient_accumulation', 16),
        learning_rate=model_config.get('learning_rate', 1e-4),
        num_train_epochs=model_config.get('epochs', 2),
        fp16=True,
        save_steps=100,
        logging_steps=10,
        warmup_ratio=model_config.get('warmup_ratio', 0.1)
    )

    # Create trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # Save model
    trainer.save_model(f'/tmp/output_{stage}')
    tokenizer.save_pretrained(f'/tmp/output_{stage}')
```

## Memory Optimization for Different Instance Types

### Local Development (4GB VRAM)
Use the original configuration from models.yaml:
```yaml
fine_tuning:
  batch_size: 1
  gradient_accumulation: 32
  quantization: "4bit"
```

### AWS g4dn.xlarge (16GB GPU Memory)
Enhanced configuration for cloud training:
```yaml
fine_tuning:
  batch_size: 4
  gradient_accumulation: 8
  quantization: "8bit"  # Can use 8bit instead of 4bit
  max_length: 512  # Increased context length
```

### AWS p3.2xlarge (16GB V100)
High-performance configuration:
```yaml
fine_tuning:
  batch_size: 8
  gradient_accumulation: 4
  quantization: "none"  # Full precision
  max_length: 1024  # Full context length
```

## Monitoring and Cost Control

### CloudWatch Metrics
Monitor the following metrics:
- **GPU Utilization**: Should be >80% during training
- **Memory Usage**: Monitor for out-of-memory errors
- **Training Loss**: Track convergence
- **Instance Count**: Monitor auto-scaling
- **Cost Per Hour**: Track spending

### Budget Alerts
Set up budget alerts at:
- **$5**: Warning for single training run
- **$10**: Critical alert for daily budget
- **$50**: Monthly budget limit

### Automated Shutdowns
Implement automatic shutdowns when:
- **Training completes** successfully
- **Budget threshold** is reached
- **High error rate** is detected
- **Spot interruption rate** exceeds 50%

## Usage Examples

### 1. Training Chemistry Models
```bash
# Set domain configuration
export DOMAIN=chemistry
export TRAINING_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/sci-tutor-training

# Start orchestrator
python src/aws_orchestrator.py
```

### 2. Training Physics Models
```bash
# Switch to physics domain
export DOMAIN=physics

# Update configuration for physics-specific data paths
python src/aws_orchestrator.py --domain physics
```

### 3. Single Model Training
```bash
# Train only the fine-tuning model
python src/aws_orchestrator.py --model-type fine_tuning --stage all
```

### 4. Cost-Optimized Training
```bash
# Use smaller instances and longer training times
python src/aws_orchestrator.py \
  --instance-type g4dn.xlarge \
  --max-spot-price 0.20 \
  --budget-limit 5.00
```

This comprehensive configuration system allows efficient training of Cite-Tutor models across different domains while maintaining cost control and resource optimization.