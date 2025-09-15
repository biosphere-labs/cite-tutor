# AWS Spot Fleet Training Orchestration for Sci-Tutor

## Overview

This document provides comprehensive documentation for training the Sci-Tutor AI models using AWS Spot Fleets for cost-effective GPU-accelerated training. The system uses spot instances to reduce training costs by up to 90% compared to on-demand instances.

## Architecture Overview

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │    │   SQS Queue     │    │  Spot Fleet     │
│   (EC2 Instance)│◄──►│  (Job Queue)    │◄──►│  (GPU Workers)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3 Bucket     │    │   CloudWatch    │    │   Auto Scaling  │
│  (Data Storage) │    │  (Monitoring)   │    │   (Fleet Mgmt)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Roles

1. **Orchestrator EC2 Instance**: Manages training pipeline, job scheduling, and fleet coordination
2. **SQS Queue**: Distributes training jobs to worker instances
3. **Spot Fleet**: GPU-enabled instances that perform the actual model training
4. **S3 Bucket**: Stores training data, models, and results
5. **CloudWatch**: Monitors system health and performance
6. **Auto Scaling**: Manages spot fleet size based on queue depth

## Infrastructure Components

### 1. Orchestrator Instance

**Purpose**: Central coordination node that manages the entire training pipeline.

**Instance Type**: `t3.medium` or `t3.large` (CPU-only, cost-effective)
**Operating System**: Ubuntu 20.04 LTS
**Location**: Any AWS region with sufficient GPU spot capacity

**Responsibilities**:
- Training job scheduling and management
- Spot fleet creation and monitoring
- Data preprocessing and model orchestration
- Result aggregation and validation
- Cost tracking and budget management

### 2. Worker Instances (Spot Fleet)

**Purpose**: GPU-enabled instances that perform model training tasks.

**Recommended Instance Types**:
- `g4dn.xlarge` (T4 GPU, 4 vCPUs, 16GB RAM) - Cost-effective for most workloads
- `g4dn.2xlarge` (T4 GPU, 8 vCPUs, 32GB RAM) - Better for larger models
- `p3.2xlarge` (V100 GPU, 8 vCPUs, 61GB RAM) - High-performance option

**Operating System**: Ubuntu 20.04 LTS with Deep Learning AMI
**Auto Scaling**: 0-10 instances based on queue depth and budget

## Setup Instructions

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with credentials
3. **Terraform** (optional, for infrastructure as code)
4. **SSH Key Pair** for EC2 access

### Step 1: Create IAM Roles and Policies

#### Orchestrator Instance Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "sqs:*",
        "s3:*",
        "cloudwatch:*",
        "logs:*",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Worker Instance Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "s3:GetObject",
        "s3:PutObject",
        "cloudwatch:PutMetricData",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### Step 2: Create Infrastructure Resources

#### 2.1 Create S3 Bucket

```bash
aws s3 mb s3://sci-tutor-training-data-[unique-id]
aws s3 mb s3://sci-tutor-models-[unique-id]
```

#### 2.2 Create SQS Queue

```bash
aws sqs create-queue --queue-name sci-tutor-training-jobs
```

#### 2.3 Create Security Groups

```bash
# Security group for orchestrator
aws ec2 create-security-group \
  --group-name sci-tutor-orchestrator \
  --description "Security group for training orchestrator"

# Allow SSH access
aws ec2 authorize-security-group-ingress \
  --group-name sci-tutor-orchestrator \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Security group for workers (more restrictive)
aws ec2 create-security-group \
  --group-name sci-tutor-workers \
  --description "Security group for spot fleet workers"
```

### Step 3: Launch Orchestrator Instance

#### 3.1 Create Orchestrator Launch Script

```bash
#!/bin/bash
# orchestrator_setup.sh

# Update system
sudo apt-get update -y
sudo apt-get install -y python3-pip git docker.io

# Install Python dependencies
pip3 install boto3 torch transformers datasets accelerate

# Clone sci-tutor repository
git clone https://github.com/your-org/sci-tutor.git
cd sci-tutor

# Install project dependencies
pip3 install -e .

# Create orchestrator service
sudo tee /etc/systemd/system/sci-tutor-orchestrator.service > /dev/null <<EOF
[Unit]
Description=Sci-Tutor Training Orchestrator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/sci-tutor
ExecStart=/usr/bin/python3 src/aws_orchestrator.py
Restart=always
RestartSec=30
Environment=AWS_DEFAULT_REGION=us-east-1

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable sci-tutor-orchestrator.service
sudo systemctl start sci-tutor-orchestrator.service
```

#### 3.2 Launch Orchestrator Instance

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --iam-instance-profile Name=sci-tutor-orchestrator-role \
  --user-data file://orchestrator_setup.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sci-tutor-orchestrator}]'
```

### Step 4: Configure Spot Fleet

#### 4.1 Create Spot Fleet Configuration

```json
{
  "SpotFleetRequestConfig": {
    "IamFleetRole": "arn:aws:iam::ACCOUNT:role/aws-ec2-spot-fleet-tagging-role",
    "AllocationStrategy": "diversified",
    "TargetCapacity": 2,
    "SpotPrice": "0.50",
    "LaunchSpecifications": [
      {
        "ImageId": "ami-0c02fb55956c7d316",
        "InstanceType": "g4dn.xlarge",
        "KeyName": "your-key-pair",
        "SecurityGroups": [
          {
            "GroupId": "sg-xxxxxxxxx"
          }
        ],
        "IamInstanceProfile": {
          "Arn": "arn:aws:iam::ACCOUNT:instance-profile/sci-tutor-worker-role"
        },
        "UserData": "base64-encoded-user-data-script",
        "TagSpecifications": [
          {
            "ResourceType": "instance",
            "Tags": [
              {
                "Key": "Name",
                "Value": "sci-tutor-worker"
              }
            ]
          }
        ]
      }
    ],
    "TerminateInstancesWithExpiration": true,
    "Type": "maintain"
  }
}
```

#### 4.2 Launch Spot Fleet

```bash
aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config.json
```

## User Data Script Explanation

The `user_data_script.sh` file runs on each spot instance when it launches. Here's what it does:

### System Setup (Lines 3-20)
1. **Updates the system** with latest packages
2. **Installs Docker** for containerized training (optional)
3. **Installs AWS CLI** for S3 and SQS access
4. **Installs Python dependencies** for ML training

### Spot Interruption Handler (Lines 22-120)
Creates a Python service that:
- **Monitors spot interruption notices** via EC2 metadata
- **Polls SQS queue** for training jobs
- **Gracefully handles shutdowns** when spot instances are reclaimed
- **Processes ML jobs** with checkpoint saving

### Service Configuration (Lines 122-142)
- **Creates systemd service** for automatic startup
- **Enables process restart** if service crashes
- **Runs as ubuntu user** for security

### Monitoring Setup (Lines 144-188)
- **Installs CloudWatch agent** for metrics and logs
- **Configures CPU, memory, and disk monitoring**
- **Sets up log collection** for debugging

## Training Workflow

### 1. Job Submission

The orchestrator breaks down training into smaller jobs:

```python
# Example job structure
training_job = {
    "job_id": "sci-tutor-stage1-001",
    "stage": "book_foundation",
    "model_config": {
        "base_model": "distilgpt2",
        "lora_r": 8,
        "batch_size": 1,
        "epochs": 2
    },
    "data_s3_path": "s3://sci-tutor-data/stage1/",
    "output_s3_path": "s3://sci-tutor-models/stage1/",
    "checkpoint_interval": 100,
    "max_runtime_minutes": 60
}
```

### 2. Worker Processing

Each worker instance:
1. **Polls SQS queue** for new jobs
2. **Downloads training data** from S3
3. **Loads model and configuration**
4. **Runs training with checkpointing**
5. **Uploads results to S3**
6. **Reports completion to orchestrator**

### 3. Interruption Handling

When spot interruption occurs:
1. **Stop accepting new jobs**
2. **Save current model checkpoint**
3. **Upload partial results to S3**
4. **Send job back to queue for retry**
5. **Graceful instance shutdown**

## Cost Optimization Strategies

### 1. Instance Selection
- **Use g4dn.xlarge** for most training tasks (T4 GPU, ~$0.10-0.30/hour spot)
- **Mix instance types** to increase availability
- **Set maximum spot price** to control costs

### 2. Training Optimization
- **Checkpoint frequently** (every 10-50 steps)
- **Use gradient accumulation** for larger effective batch sizes
- **Implement mixed precision** training
- **Split large training runs** into smaller jobs

### 3. Auto Scaling
Configure auto scaling based on:
- **Queue depth** (scale up when >5 jobs waiting)
- **Time of day** (scale down during peak pricing hours)
- **Budget remaining** (scale down when approaching limits)

### 4. Monitoring and Alerts

Set up CloudWatch alarms for:
- **High spot interruption rates**
- **Training job failures**
- **Cost exceeding budget**
- **Queue depth growth**

## Security Considerations

### 1. IAM Permissions
- **Use least privilege principle**
- **Separate roles for orchestrator and workers**
- **Enable CloudTrail** for audit logging

### 2. Network Security
- **Use private subnets** for worker instances
- **Restrict security group rules**
- **Enable VPC flow logs**

### 3. Data Protection
- **Encrypt S3 buckets** with KMS
- **Use HTTPS for all API calls**
- **Rotate access keys regularly**

## Troubleshooting

### Common Issues

#### 1. Spot Instances Not Launching
- **Check spot pricing** in your region
- **Verify instance type availability**
- **Review IAM permissions**
- **Check security group rules**

#### 2. Training Jobs Failing
- **Check CloudWatch logs** for error messages
- **Verify S3 bucket permissions**
- **Monitor GPU memory usage**
- **Check for CUDA compatibility**

#### 3. High Interruption Rates
- **Increase spot price limit**
- **Use diversified instance types**
- **Consider different availability zones**
- **Adjust auto scaling policies**

### Monitoring Commands

```bash
# Check orchestrator status
ssh ubuntu@orchestrator-ip 'sudo systemctl status sci-tutor-orchestrator'

# View worker logs
aws logs describe-log-streams --log-group-name gpu-spot-fleet

# Check SQS queue depth
aws sqs get-queue-attributes --queue-url YOUR_QUEUE_URL --attribute-names ApproximateNumberOfMessages

# Monitor spot fleet
aws ec2 describe-spot-fleet-requests
```

## Cost Estimation

### Example Training Costs

For a typical 3-stage training pipeline:

| Stage | Duration | Instance Type | Spot Price | Cost |
|-------|----------|---------------|------------|------|
| Stage 1 | 2 hours | g4dn.xlarge | $0.15/hr | $0.30 |
| Stage 2 | 1 hour | g4dn.xlarge | $0.15/hr | $0.15 |
| Stage 3 | 1 hour | g4dn.xlarge | $0.15/hr | $0.15 |
| **Total** | **4 hours** | - | - | **$0.60** |

### Additional Costs
- **Orchestrator instance**: $0.02/hour (t3.medium)
- **S3 storage**: ~$0.01/GB/month
- **Data transfer**: Minimal (same region)
- **CloudWatch**: ~$0.50/month for basic monitoring

**Total estimated cost for complete training pipeline: Under $1.00**

## Next Steps

1. **Set up development environment** with smaller instances
2. **Test training pipeline** with sample data
3. **Implement monitoring dashboards**
4. **Create automated deployment scripts**
5. **Set up budget alerts and controls**

This AWS spot fleet orchestration system provides a cost-effective, scalable solution for training Sci-Tutor models while maintaining reliability through proper interruption handling and checkpointing.