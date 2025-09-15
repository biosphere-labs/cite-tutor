# AWS Spot Fleet Training Setup Guide for Sci-Tutor

## What This Guide Covers

This guide explains how to train the models defined in `config/models.yaml` using AWS Spot Fleet. The models.yaml file defines 4 types of models that need training:

1. **Structure Analysis Model** - Analyzes document structure
2. **Q&A Generation Model** - Creates question-answer pairs
3. **Fine-Tuning Model** - Main conversational AI (3 stages)
4. **Embedding Model** - For document search/retrieval

## Overview of the Training System

The system uses two types of EC2 instances:
- **Orchestrator Instance** (t3.medium): Manages jobs and coordinates training
- **Worker Instances** (g4dn.xlarge): GPU instances that do the actual training

```
Orchestrator → SQS Queue → Spot Fleet Workers → S3 Storage
    (CPU)        (Jobs)      (GPU Training)     (Models)
```

## Step 1: Prerequisites Setup

### 1.1 AWS Account Setup
- AWS account with billing enabled
- AWS CLI installed and configured
- EC2 key pair created for SSH access

### 1.2 Create Required AWS Resources

**Create S3 Buckets:**
```bash
aws s3 mb s3://sci-tutor-training-data-$(date +%s)
aws s3 mb s3://sci-tutor-models-$(date +%s)
```
*Note: Replace $(date +%s) with a unique identifier*

**Create SQS Queue:**
```bash
aws sqs create-queue --queue-name sci-tutor-training-jobs
```

**Create IAM Roles:**
1. Create role: `sci-tutor-orchestrator-role`
2. Create role: `sci-tutor-worker-role`
3. Create role: `aws-ec2-spot-fleet-tagging-role`

## Step 2: Launch Orchestrator Instance

### 2.1 Create Orchestrator Instance

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name YOUR-KEY-PAIR \
  --security-group-ids sg-YOUR-SECURITY-GROUP \
  --iam-instance-profile Name=sci-tutor-orchestrator-role \
  --user-data file://orchestrator_setup.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sci-tutor-orchestrator}]'
```

### 2.2 Orchestrator Setup Script (orchestrator_setup.sh)

```bash
#!/bin/bash
# Update system
sudo apt-get update -y
sudo apt-get install -y python3-pip git

# Install dependencies
pip3 install boto3 torch transformers datasets accelerate peft

# Clone repository
git clone YOUR-REPO-URL sci-tutor
cd sci-tutor

# Set environment variables
echo 'export SQS_QUEUE_URL="YOUR-SQS-QUEUE-URL"' >> ~/.bashrc
echo 'export S3_DATA_BUCKET="YOUR-DATA-BUCKET"' >> ~/.bashrc
echo 'export S3_MODEL_BUCKET="YOUR-MODEL-BUCKET"' >> ~/.bashrc

# Start orchestrator service
python3 src/training_orchestrator.py
```

## Step 3: Understanding the Models from models.yaml

The `config/models.yaml` file defines these models:

### Structure Analysis Model
```yaml
structure_analysis:
  model: "google/flan-t5-small"  # 60MB
  max_memory_mb: 1024
```
**Purpose**: Extracts sections from academic papers
**Training Time**: ~30 minutes on g4dn.xlarge
**Cost**: ~$0.08

### Q&A Generation Model
```yaml
qa_generation:
  model: "google/flan-t5-small"
  batch_size: 1
  max_length: 128
```
**Purpose**: Creates question-answer pairs from content
**Training Time**: ~45 minutes on g4dn.xlarge
**Cost**: ~$0.12

### Fine-Tuning Model (Main Model)
```yaml
fine_tuning:
  base_model: "distilgpt2"  # 82MB
  quantization: "4bit"
  lora_r: 16
  lora_alpha: 32
  batch_size: 1
  gradient_accumulation: 32
```
**Purpose**: Main conversational AI for tutoring
**Training Stages**: 3 stages (Book → Paper → Integration)
**Training Time**: ~4 hours total on g4dn.xlarge
**Cost**: ~$0.60

### Embedding Model
```yaml
embedding:
  model: "all-MiniLM-L6-v2"  # 22MB
```
**Purpose**: Document search and retrieval
**Training Time**: ~30 minutes on g4dn.xlarge
**Cost**: ~$0.08

**Total Training Cost: ~$0.88**

## Step 4: How the user_data_script.sh Works

The `user_data_script.sh` runs on each GPU worker instance when it starts. Here's what it does:

### 4.1 System Setup (Lines 1-20)
```bash
# Updates Ubuntu
apt-get update -y

# Installs Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Installs AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Installs Python ML libraries
pip3 install boto3 torch transformers accelerate
```

### 4.2 Creates Job Processing Service (Lines 22-120)
Creates a Python script that:
- Polls SQS queue for training jobs
- Downloads training data from S3
- Runs the specific model training based on models.yaml config
- Saves results back to S3
- Handles spot instance interruptions gracefully

### 4.3 Creates System Service (Lines 122-142)
```bash
# Creates systemd service for automatic startup
cat > /etc/systemd/system/ml-worker.service << 'EOF'
[Unit]
Description=ML Job Worker
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/usr/bin/python3 /home/ubuntu/spot_handler.py
Restart=always
RestartSec=10
EOF

# Enables and starts the service
systemctl enable ml-worker.service
systemctl start ml-worker.service
```

### 4.4 Sets Up Monitoring (Lines 144-188)
Installs CloudWatch agent to monitor:
- CPU usage
- Memory usage
- Disk usage
- Training logs

## Step 5: Launch Spot Fleet for Training

### 5.1 Prepare User Data Script
First, update the user_data_script.sh with your SQS queue URL:
```bash
# In user_data_script.sh, line 117:
queue_url = "YOUR-ACTUAL-SQS-QUEUE-URL"
```

### 5.2 Create Spot Fleet Configuration
```json
{
  "SpotFleetRequestConfig": {
    "IamFleetRole": "arn:aws:iam::YOUR-ACCOUNT:role/aws-ec2-spot-fleet-tagging-role",
    "AllocationStrategy": "diversified",
    "TargetCapacity": 2,
    "SpotPrice": "0.30",
    "LaunchSpecifications": [
      {
        "ImageId": "ami-0c02fb55956c7d316",
        "InstanceType": "g4dn.xlarge",
        "KeyName": "YOUR-KEY-PAIR",
        "SecurityGroups": [{"GroupId": "YOUR-SECURITY-GROUP"}],
        "IamInstanceProfile": {
          "Arn": "arn:aws:iam::YOUR-ACCOUNT:instance-profile/sci-tutor-worker-role"
        },
        "UserData": "BASE64-ENCODED-USER-DATA-SCRIPT"
      }
    ],
    "TerminateInstancesWithExpiration": true,
    "Type": "maintain"
  }
}
```

### 5.3 Encode User Data Script
```bash
# Convert user_data_script.sh to base64
base64 -w 0 user_data_script.sh > user_data_encoded.txt
```

Replace "BASE64-ENCODED-USER-DATA-SCRIPT" in the JSON with the contents of user_data_encoded.txt.

### 5.4 Launch Spot Fleet
```bash
aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config.json
```

## Step 6: Start Training Process

### 6.1 SSH into Orchestrator Instance
```bash
ssh -i YOUR-KEY.pem ubuntu@ORCHESTRATOR-IP
```

### 6.2 Upload Training Data to S3
```bash
# Upload your training data
aws s3 cp data/ s3://YOUR-DATA-BUCKET/chemistry/ --recursive
```

### 6.3 Start Training Pipeline
```bash
cd sci-tutor
python3 src/training_orchestrator.py --domain chemistry
```

This will create training jobs for all 4 model types and submit them to the SQS queue.

## Step 7: Monitor Training Progress

### 7.1 Check Queue Status
```bash
aws sqs get-queue-attributes \
  --queue-url YOUR-QUEUE-URL \
  --attribute-names ApproximateNumberOfMessages
```

### 7.2 Check Spot Fleet Status
```bash
aws ec2 describe-spot-fleet-requests
```

### 7.3 Monitor Logs
```bash
# View CloudWatch logs
aws logs describe-log-streams --log-group-name gpu-spot-fleet
```

### 7.4 Check Training Results
```bash
# List completed models
aws s3 ls s3://YOUR-MODEL-BUCKET/chemistry/
```

## Step 8: Training Sequence

The models train in this order:

1. **Structure Analysis** (30 min) - Processes documents
2. **Q&A Generation** (45 min) - Creates training data
3. **Fine-Tuning Stage 1** (2 hours) - Book knowledge
4. **Fine-Tuning Stage 2** (1 hour) - Paper integration
5. **Fine-Tuning Stage 3** (1 hour) - Knowledge synthesis
6. **Embedding Model** (30 min) - Document search

**Total Time: ~5.5 hours**
**Total Cost: ~$0.88 (at $0.16/hour spot price)**

## Step 9: What Happens During Training

### On Worker Instances (GPU):
1. Worker starts and polls SQS queue
2. Receives training job message
3. Downloads model config from models.yaml settings
4. Downloads training data from S3
5. Loads the specific model (flan-t5-small, distilgpt2, etc.)
6. Applies optimizations (4-bit quantization, LoRA, etc.)
7. Runs training with the configured parameters
8. Saves trained model to S3
9. Reports completion and gets next job

### On Orchestrator Instance (CPU):
1. Reads models.yaml configuration
2. Creates training jobs for each model type
3. Submits jobs to SQS queue in correct order
4. Monitors spot fleet scaling
5. Tracks job completion
6. Manages budget and cost controls

## Step 10: Cost Control

### Budget Monitoring:
- Set up billing alerts at $5, $10, $50
- Monitor spot prices (should be $0.10-0.30/hour)
- Auto-shutdown when budget reached

### Cost Optimization:
- Use diversified instance types
- Set maximum spot price limits
- Scale down fleet when queue empty
- Use 4-bit quantization to reduce memory needs

## Troubleshooting Common Issues

### Issue: Spot instances not launching
**Solution**: Check spot pricing and increase max price

### Issue: Training jobs failing
**Solution**: Check CloudWatch logs for CUDA/memory errors

### Issue: High costs
**Solution**: Verify auto-scaling policies and spot price limits

### Issue: Workers not processing jobs
**Solution**: Verify SQS queue URL in user_data_script.sh

## Next Steps After Training

1. **Download trained models** from S3
2. **Test model performance** with validation data
3. **Deploy models** for inference
4. **Clean up resources** to stop costs

The trained models will be saved in S3 at:
- `s3://YOUR-MODEL-BUCKET/chemistry/structure/`
- `s3://YOUR-MODEL-BUCKET/chemistry/qa/`
- `s3://YOUR-MODEL-BUCKET/chemistry/final/`
- `s3://YOUR-MODEL-BUCKET/chemistry/embedding/`

This setup provides cost-effective training of all Sci-Tutor models using AWS Spot Fleet with automatic scaling and interruption handling.