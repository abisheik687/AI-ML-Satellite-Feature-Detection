# Deployment Package and Installation Guide

## Quick Start

### System Requirements
- **Operating System:** Ubuntu 20.04+ or CentOS 8+
- **Python:** 3.8 or higher
- **GPU:** NVIDIA GPU with CUDA 11.8+ (recommended)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 100GB free space minimum
- **Network:** High-speed internet connection for data download

### One-Command Installation
```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/your-org/geospatial-pipeline/main/install.sh | bash
```

## Manual Installation

### 1. Environment Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y gdal-bin libgdal-dev
sudo apt install -y git wget curl unzip

# Install NVIDIA drivers and CUDA (if using GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda
```

### 2. Clone Repository
```bash
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline
```

### 3. Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional geospatial tools
pip install whitebox whiteboxgui
```

### 4. Download Pre-trained Models
```bash
# Create models directory
mkdir -p models

# Download pre-trained models
wget -O models/glacial_lake_unet.pth "https://your-model-storage.com/glacial_lake_unet.pth"
wget -O models/road_deeplabv3.pth "https://your-model-storage.com/road_deeplabv3.pth"
wget -O models/drainage_rf.pkl "https://your-model-storage.com/drainage_rf.pkl"
```

### 5. Configuration
```bash
# Copy configuration template
cp config/config.template.yaml config/config.yaml

# Edit configuration file
nano config/config.yaml
```

## Docker Deployment

### 1. Using Pre-built Image
```bash
# Pull the latest image
docker pull your-registry/geospatial-pipeline:latest

# Run container
docker run -d \
  --name geospatial-pipeline \
  --gpus all \
  -p 8080:8080 \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  your-registry/geospatial-pipeline:latest
```

### 2. Building from Source
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models
RUN mkdir -p models && \
    wget -O models/glacial_lake_unet.pth "https://your-model-storage.com/glacial_lake_unet.pth" && \
    wget -O models/road_deeplabv3.pth "https://your-model-storage.com/road_deeplabv3.pth" && \
    wget -O models/drainage_rf.pkl "https://your-model-storage.com/drainage_rf.pkl"

# Expose ports
EXPOSE 8080 5000

# Start services
CMD ["python3", "start_services.py"]
```

```bash
# Build image
docker build -t geospatial-pipeline .

# Run container
docker run -d \
  --name geospatial-pipeline \
  --gpus all \
  -p 8080:8080 \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  geospatial-pipeline
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: geospatial-pipeline

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pipeline-config
  namespace: geospatial-pipeline
data:
  config.yaml: |
    models:
      glacial_lakes: "/app/models/glacial_lake_unet.pth"
      roads: "/app/models/road_deeplabv3.pth"
      drainage: "/app/models/drainage_rf.pkl"
    
    processing:
      batch_size: 4
      max_workers: 2
      gpu_enabled: true
    
    api:
      host: "0.0.0.0"
      port: 5000
      max_file_size: "100MB"
    
    dashboard:
      host: "0.0.0.0"
      port: 8080
```

### 2. Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: geospatial-pipeline
  namespace: geospatial-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: geospatial-pipeline
  template:
    metadata:
      labels:
        app: geospatial-pipeline
    spec:
      containers:
      - name: pipeline
        image: your-registry/geospatial-pipeline:latest
        ports:
        - containerPort: 5000
          name: api
        - containerPort: 8080
          name: dashboard
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
        - name: output
          mountPath: /app/output
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
      volumes:
      - name: config
        configMap:
          name: pipeline-config
      - name: data
        persistentVolumeClaim:
          claimName: data-pvc
      - name: output
        persistentVolumeClaim:
          claimName: output-pvc
```

### 3. Services and Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pipeline-service
  namespace: geospatial-pipeline
spec:
  selector:
    app: geospatial-pipeline
  ports:
  - name: api
    port: 5000
    targetPort: 5000
  - name: dashboard
    port: 8080
    targetPort: 8080
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pipeline-ingress
  namespace: geospatial-pipeline
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  rules:
  - host: geospatial-pipeline.your-domain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: pipeline-service
            port:
              number: 5000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pipeline-service
            port:
              number: 8080
```

## Cloud Deployment

### AWS Deployment
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure

# Deploy using CloudFormation
aws cloudformation create-stack \
  --stack-name geospatial-pipeline \
  --template-body file://aws/cloudformation.yaml \
  --parameters ParameterKey=InstanceType,ParameterValue=p3.2xlarge \
  --capabilities CAPABILITY_IAM
```

### Google Cloud Deployment
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Deploy to Google Kubernetes Engine
gcloud container clusters create geospatial-cluster \
  --num-nodes=2 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-k80,count=1 \
  --zone=us-central1-a

# Deploy application
kubectl apply -f k8s/
```

### Azure Deployment
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create resource group
az group create --name geospatial-rg --location eastus

# Deploy using ARM template
az deployment group create \
  --resource-group geospatial-rg \
  --template-file azure/template.json \
  --parameters @azure/parameters.json
```

## Configuration

### Main Configuration File (config.yaml)
```yaml
# Application settings
app:
  name: "Geospatial Analysis Pipeline"
  version: "1.0.0"
  debug: false

# Model settings
models:
  glacial_lakes:
    path: "models/glacial_lake_unet.pth"
    device: "cuda"
    batch_size: 4
    threshold: 0.5
    min_area: 1000  # square meters
  
  roads:
    path: "models/road_deeplabv3.pth"
    device: "cuda"
    batch_size: 4
    threshold: 0.5
    spur_length: 10  # pixels
  
  drainage:
    path: "models/drainage_rf.pkl"
    flow_threshold: 1000
    min_stream_length: 100  # meters

# Processing settings
processing:
  max_workers: 4
  chunk_size: 1024  # pixels
  overlap: 128  # pixels
  output_format: "geojson"  # or "shapefile"
  
# Data sources
data_sources:
  sentinel2:
    enabled: true
    cloud_cover_threshold: 20
  sentinel1:
    enabled: true
    polarization: ["VV", "VH"]
  dem:
    source: "SRTM"  # or "CartoDEM", "ASTER"
    resolution: 30  # meters

# API settings
api:
  host: "0.0.0.0"
  port: 5000
  max_file_size: "100MB"
  rate_limit: "100/hour"
  cors_origins: ["*"]

# Dashboard settings
dashboard:
  host: "0.0.0.0"
  port: 8080
  title: "Geospatial Analysis Pipeline"
  map_center: [28.2380, 83.9956]
  map_zoom: 8

# Database settings (optional)
database:
  enabled: false
  type: "postgresql"
  host: "localhost"
  port: 5432
  name: "geospatial_db"
  user: "postgres"
  password: "password"

# Logging settings
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"
  max_size: "10MB"
  backup_count: 5

# Monitoring settings
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30  # seconds
```

### Environment Variables
```bash
# .env file
PIPELINE_CONFIG_PATH=/app/config/config.yaml
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=/app
LOG_LEVEL=INFO

# Google Earth Engine (if using)
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gee-service-account.json

# AWS (if using)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Database (if using)
DATABASE_URL=postgresql://user:password@localhost:5432/geospatial_db
```

## Service Management

### Systemd Service (Linux)
```ini
# /etc/systemd/system/geospatial-pipeline.service
[Unit]
Description=Geospatial Analysis Pipeline
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/geospatial-pipeline
Environment=PATH=/opt/geospatial-pipeline/venv/bin
ExecStart=/opt/geospatial-pipeline/venv/bin/python start_services.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable geospatial-pipeline
sudo systemctl start geospatial-pipeline
sudo systemctl status geospatial-pipeline
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  pipeline:
    build: .
    ports:
      - "8080:8080"
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./config:/app/config
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  postgres:
    image: postgis/postgis:13-3.1
    environment:
      POSTGRES_DB: geospatial_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

## Monitoring and Maintenance

### Health Checks
```python
# health_check.py
import requests
import sys

def check_api_health():
    try:
        response = requests.get('http://localhost:5000/api/status', timeout=10)
        return response.status_code == 200
    except:
        return False

def check_dashboard_health():
    try:
        response = requests.get('http://localhost:8080', timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    api_healthy = check_api_health()
    dashboard_healthy = check_dashboard_health()
    
    if api_healthy and dashboard_healthy:
        print("All services healthy")
        sys.exit(0)
    else:
        print("Some services unhealthy")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Monitoring Script
```bash
#!/bin/bash
# monitor.sh

# Check disk space
DISK_USAGE=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEM_USAGE -gt 80 ]; then
    echo "WARNING: Memory usage is ${MEM_USAGE}%"
fi

# Check GPU usage (if available)
if command -v nvidia-smi &> /dev/null; then
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    echo "GPU usage: ${GPU_USAGE}%"
fi

# Check service status
python3 health_check.py
```

### Log Rotation
```bash
# /etc/logrotate.d/geospatial-pipeline
/opt/geospatial-pipeline/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        systemctl reload geospatial-pipeline
    endscript
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.yaml
   models:
     glacial_lakes:
       batch_size: 2  # Reduce from 4
   ```

2. **GDAL Import Error**
   ```bash
   # Install GDAL development headers
   sudo apt install libgdal-dev
   pip install --upgrade gdal==$(gdal-config --version)
   ```

3. **Model Loading Error**
   ```bash
   # Check model file exists and permissions
   ls -la models/
   # Re-download models if corrupted
   ./scripts/download_models.sh
   ```

4. **Port Already in Use**
   ```bash
   # Find process using port
   sudo lsof -i :5000
   # Kill process or change port in config
   ```

### Performance Optimization

1. **GPU Memory Optimization**
   ```python
   # Add to model loading code
   torch.cuda.empty_cache()
   torch.backends.cudnn.benchmark = True
   ```

2. **CPU Optimization**
   ```bash
   # Set environment variables
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

3. **I/O Optimization**
   ```bash
   # Use SSD for temporary files
   export TMPDIR=/fast/ssd/tmp
   ```

## Security Considerations

### API Security
```python
# Add to Flask app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/process/glacial-lakes', methods=['POST'])
@limiter.limit("10 per minute")
def process_glacial_lakes():
    # Implementation
    pass
```

### File Upload Security
```python
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# In upload handler
if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    # Process file
```

### Network Security
```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 5000/tcp  # API (if needed)
sudo ufw allow 8080/tcp  # Dashboard (if needed)
```

This deployment package provides comprehensive instructions for installing, configuring, and maintaining the geospatial analysis pipeline across various environments and platforms.

