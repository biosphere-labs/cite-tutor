#!/bin/bash

# Update system
apt-get update -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
fi

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install Python dependencies
pip3 install boto3 torch transformers accelerate

# Setup spot interruption handler
cat > /home/ubuntu/spot_handler.py << 'EOF'
import boto3
import requests
import time
import sys
import signal
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotInterruptionHandler:
    def __init__(self, sqs_queue_url):
        self.sqs = boto3.client('sqs')
        self.queue_url = sqs_queue_url
        self.running = True
        self.current_job = None
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGINT, self.graceful_shutdown)
    
    def check_spot_interruption(self):
        """Check if spot interruption notice has been issued"""
        try:
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/spot/instance-action',
                timeout=2
            )
            if response.status_code == 200:
                logger.warning("Spot interruption notice received!")
                return True
        except requests.RequestException:
            pass  # No interruption notice
        return False
    
    def graceful_shutdown(self, signum, frame):
        logger.info(f"Received signal {signum}, starting graceful shutdown...")
        self.running = False
    
    def process_jobs(self):
        while self.running:
            # Check for spot interruption every 30 seconds
            if self.check_spot_interruption():
                logger.warning("Spot interruption detected, stopping job processing")
                break
            
            try:
                # Poll SQS for messages
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20
                )
                
                messages = response.get('Messages', [])
                if not messages:
                    continue
                
                message = messages[0]
                self.current_job = message
                
                # Process the job here
                self.process_ml_job(message['Body'])
                
                # Delete message after successful processing
                self.sqs.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
                
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Error processing job: {e}")
                time.sleep(5)
    
    def process_ml_job(self, job_data):
        """Process ML job - replace with your actual ML logic"""
        import json
        job = json.loads(job_data)
        
        logger.info(f"Processing job: {job.get('job_id', 'unknown')}")

        # Your ML model training code here
        # Example: Cite-Tutor model training
        # model = load_model()
        # result = model.train(job['training_data'])
        # save_model_to_s3(result, job['output_path'])
        
        time.sleep(10)  # Simulate processing time
        logger.info("Job completed successfully")

if __name__ == "__main__":
    queue_url = "${sqs_queue_url}"
    handler = SpotInterruptionHandler(queue_url)
    handler.process_jobs()
EOF

# Create systemd service for the job processor
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

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl enable ml-worker.service
systemctl start ml-worker.service

# Setup CloudWatch agent for monitoring
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Create basic CloudWatch config
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
    "metrics": {
        "namespace": "GPU-Spot-Fleet",
        "metrics_collected": {
            "cpu": {
                "measurement": ["cpu_usage_idle", "cpu_usage_iowait", "cpu_usage_user", "cpu_usage_system"],
                "metrics_collection_interval": 60,
                "totalcpu": false
            },
            "disk": {
                "measurement": ["used_percent"],
                "metrics_collection_interval": 60,
                "resources": ["*"]
            },
            "mem": {
                "measurement": ["mem_used_percent"],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/ml-worker.log",
                        "log_group_name": "gpu-spot-fleet",
                        "log_stream_name": "{instance_id}-ml-worker"
                    }
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s