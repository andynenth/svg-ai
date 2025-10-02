# 🚀 EC2 Budget Deployment Checklist for SVG-AI

## 💰 Cost Breakdown
- **EC2 t3.micro**: $0/month (Free tier 1st year, then $8/month)
- **S3 + CloudFront**: ~$5/month
- **Domain (Route53)**: $12/year
- **Total Monthly**: $5 first year, $13 after
- **85% cheaper than ECS!**

## 🔑 AWS Account Details
- **Account ID**: 300079938592
- **IAM User**: Andy
- **Region**: us-east-1
- **Free Tier Status**: Check at https://console.aws.amazon.com/billing/

## 📋 Phase 0: Pre-Flight Checks (15 mins)

### Verify What You Have
- [x] AWS Account (300079938592)
- [x] AWS CLI configured
- [ ] Check free tier eligibility: `aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-12-31 --granularity MONTHLY --metrics UsageQuantity --group-by Type=DIMENSION,Key=SERVICE | grep EC2`
- [ ] Working backend locally: `python -m backend.app`
- [ ] Models exist: `ls -lh models/production/` (should show ~7MB total)
- [ ] Git repo clean: `git status`
- [ ] Create deployment branch: `git checkout -b production-deploy`

## 🏗️ Phase 1: Simple Infrastructure (1 hour)

### 1.1 Create Security Group (10 mins)
- [ ] Create security group:
```bash
aws ec2 create-security-group \
  --group-name svg-ai-sg \
  --description "SVG-AI Application Security Group" \
  --output json | tee sg-creation.json
```
- [ ] Save Security Group ID: `export SG_ID=$(cat sg-creation.json | grep GroupId | cut -d'"' -f4)`
- [ ] Add SSH access (your IP only):
```bash
export MY_IP=$(curl -s https://api.ipify.org)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr $MY_IP/32
```
- [ ] Add HTTP access (all):
```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0
```
- [ ] Add HTTPS access (all):
```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```
- [ ] Add API port (8001) temporarily:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8001 \
  --cidr 0.0.0.0/0
```
- [ ] Document SG ID in `.env.production`: `echo "SG_ID=$SG_ID" >> .env.production`

### 1.2 Create Key Pair (5 mins)
- [ ] Generate key pair:
```bash
aws ec2 create-key-pair \
  --key-name svg-ai-key \
  --query 'KeyMaterial' \
  --output text > svg-ai-key.pem
```
- [ ] Set permissions: `chmod 400 svg-ai-key.pem`
- [ ] Move to safe location: `mv svg-ai-key.pem ~/.ssh/`
- [ ] Test key file exists: `ls -la ~/.ssh/svg-ai-key.pem`
- [ ] Add to SSH config:
```bash
echo "Host svg-ai
  HostName <EC2_IP_LATER>
  User ubuntu
  IdentityFile ~/.ssh/svg-ai-key.pem" >> ~/.ssh/config
```

### 1.3 Launch EC2 Instance (10 mins)
- [ ] Get latest Ubuntu AMI ID:
```bash
export AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query 'Images[0].ImageId' \
  --output text)
echo "AMI ID: $AMI_ID"
```
- [ ] Launch t3.micro instance:
```bash
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.micro \
  --key-name svg-ai-key \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=svg-ai-server}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
  --output json | tee instance-creation.json
```
- [ ] Save Instance ID: `export INSTANCE_ID=$(cat instance-creation.json | grep InstanceId | head -1 | cut -d'"' -f4)`
- [ ] Wait for instance to run:
```bash
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
echo "Instance is running!"
```
- [ ] Get public IP:
```bash
export EC2_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)
echo "EC2 Public IP: $EC2_IP"
```
- [ ] Update SSH config with IP: `sed -i "" "s/<EC2_IP_LATER>/$EC2_IP/" ~/.ssh/config`
- [ ] Document in `.env.production`:
```bash
echo "INSTANCE_ID=$INSTANCE_ID" >> .env.production
echo "EC2_IP=$EC2_IP" >> .env.production
```
- [ ] Test SSH connection: `ssh svg-ai "echo 'Connected successfully!'"`

### 1.4 Create S3 Buckets (10 mins)
- [ ] Generate unique suffix: `export BUCKET_SUFFIX=$(date +%s)`
- [ ] Create frontend bucket:
```bash
aws s3 mb s3://svg-ai-frontend-$BUCKET_SUFFIX
echo "FRONTEND_BUCKET=svg-ai-frontend-$BUCKET_SUFFIX" >> .env.production
```
- [ ] Create models bucket:
```bash
aws s3 mb s3://svg-ai-models-$BUCKET_SUFFIX
echo "MODELS_BUCKET=svg-ai-models-$BUCKET_SUFFIX" >> .env.production
```
- [ ] Enable static website on frontend bucket:
```bash
aws s3 website s3://svg-ai-frontend-$BUCKET_SUFFIX \
  --index-document index.html \
  --error-document error.html
```
- [ ] Create bucket policy file `bucket-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::svg-ai-frontend-SUFFIX/*"
  }]
}
```
- [ ] Replace SUFFIX in policy: `sed -i "" "s/SUFFIX/$BUCKET_SUFFIX/" bucket-policy.json`
- [ ] Apply bucket policy:
```bash
aws s3api put-bucket-policy \
  --bucket svg-ai-frontend-$BUCKET_SUFFIX \
  --policy file://bucket-policy.json
```
- [ ] Upload models:
```bash
aws s3 sync models/production/ s3://svg-ai-models-$BUCKET_SUFFIX/
```
- [ ] Set public read on models:
```bash
aws s3api put-object-acl \
  --bucket svg-ai-models-$BUCKET_SUFFIX \
  --key logo_classifier.torchscript \
  --acl public-read
```

## 🖥️ Phase 2: Server Setup (45 mins)

### 2.1 Initial Server Configuration (15 mins)
- [ ] SSH into server: `ssh svg-ai`
- [ ] Update system:
```bash
sudo apt update && sudo apt upgrade -y
```
- [ ] Install Python 3.9 and essentials:
```bash
sudo apt install -y python3.9 python3.9-venv python3.9-dev python3-pip
sudo apt install -y nginx git build-essential
sudo apt install -y libpq-dev python3-wheel
```
- [ ] Install system dependencies for OpenCV and image processing:
```bash
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
sudo apt install -y libglib2.0-0 libgl1-mesa-glx
```
- [ ] Create application user:
```bash
sudo useradd -m -s /bin/bash svgai
sudo usermod -aG sudo svgai
```
- [ ] Create app directory:
```bash
sudo mkdir -p /opt/svg-ai
sudo chown svgai:svgai /opt/svg-ai
```
- [ ] Switch to app user: `sudo su - svgai`
- [ ] Create Python virtual environment:
```bash
cd /opt/svg-ai
python3.9 -m venv venv
source venv/bin/activate
```

### 2.2 Deploy Application Code (15 mins)
- [ ] Exit to ubuntu user: `exit`
- [ ] Create deployment archive locally:
```bash
# On your local machine
tar -czf svg-ai-deploy.tar.gz \
  --exclude='.git' \
  --exclude='venv*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data/logos' \
  --exclude='uploads' \
  backend frontend models requirements*.txt *.py
```
- [ ] Copy to server:
```bash
scp svg-ai-deploy.tar.gz svg-ai:/tmp/
```
- [ ] Extract on server:
```bash
ssh svg-ai
sudo su - svgai
cd /opt/svg-ai
tar -xzf /tmp/svg-ai-deploy.tar.gz
```
- [ ] Install Python dependencies:
```bash
source venv/bin/activate
pip install --upgrade pip
pip install wheel
```
- [ ] Install requirements (without AI for faster setup):
```bash
pip install -r requirements.txt
```
- [ ] Install VTracer:
```bash
export TMPDIR=/tmp
pip install vtracer
```
- [ ] Test import:
```bash
python -c "import vtracer; print('VTracer OK')"
python -c "from backend import app; print('Backend OK')"
```

### 2.3 Configure Application (10 mins)
- [ ] Create environment file:
```bash
cat > /opt/svg-ai/.env << 'EOF'
FLASK_ENV=production
MODEL_DIR=/opt/svg-ai/models/production
UPLOAD_DIR=/tmp/uploads
MAX_FILE_SIZE_MB=10
CACHE_DIR=/tmp/cache
LOG_LEVEL=INFO
EOF
```
- [ ] Create necessary directories:
```bash
mkdir -p /tmp/uploads /tmp/cache
chmod 777 /tmp/uploads /tmp/cache
```
- [ ] Create systemd service file:
```bash
sudo tee /etc/systemd/system/svg-ai.service << 'EOF'
[Unit]
Description=SVG-AI API Service
After=network.target

[Service]
Type=simple
User=svgai
WorkingDirectory=/opt/svg-ai
Environment="PATH=/opt/svg-ai/venv/bin"
ExecStart=/opt/svg-ai/venv/bin/python -m backend.app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```
- [ ] Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable svg-ai
sudo systemctl start svg-ai
```
- [ ] Check service status: `sudo systemctl status svg-ai`
- [ ] Test API locally on server: `curl http://localhost:8001/health`
- [ ] Check logs if needed: `sudo journalctl -u svg-ai -f`

### 2.4 Configure Nginx (10 mins)
- [ ] Create Nginx config:
```bash
sudo tee /etc/nginx/sites-available/svg-ai << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 20M;

    location /api/ {
        proxy_pass http://127.0.0.1:8001/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }

    location /health {
        proxy_pass http://127.0.0.1:8001/health;
    }

    location / {
        return 200 'SVG-AI API Server\n';
        add_header Content-Type text/plain;
    }
}
EOF
```
- [ ] Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/svg-ai /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
```
- [ ] Test Nginx config: `sudo nginx -t`
- [ ] Restart Nginx: `sudo systemctl restart nginx`
- [ ] Test from local machine: `curl http://$EC2_IP/health`
- [ ] Test API endpoint: `curl http://$EC2_IP/api/`
- [ ] Remove port 8001 from security group (security):
```bash
# On local machine
aws ec2 revoke-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8001 \
  --cidr 0.0.0.0/0
```

## 🎨 Phase 3: Frontend Deployment (30 mins)

### 3.1 Prepare Frontend (10 mins)
- [ ] Update API endpoint in frontend locally:
```bash
# On local machine
sed -i "" "s|http://localhost:8001|http://$EC2_IP|g" frontend/js/modules/converter.js
sed -i "" "s|localhost:8001|$EC2_IP|g" frontend/js/modules/*.js
```
- [ ] Update CORS in backend to allow S3 bucket:
```bash
echo "# Note: Update CORS in backend/app.py to allow your S3 bucket URL" >> deployment-notes.txt
```
- [ ] Test frontend locally with production API:
```bash
python -m http.server 8080
# Open http://localhost:8080/frontend/ and test
```
- [ ] Create production frontend config:
```bash
cat > frontend/config.js << EOF
window.API_BASE_URL = 'http://$EC2_IP';
window.API_TIMEOUT = 30000;
EOF
```

### 3.2 Deploy to S3 (10 mins)
- [ ] Upload frontend to S3:
```bash
aws s3 sync frontend/ s3://svg-ai-frontend-$BUCKET_SUFFIX/ \
  --exclude ".git/*" \
  --exclude "*.md" \
  --exclude ".DS_Store"
```
- [ ] Set cache headers for assets:
```bash
aws s3 cp s3://svg-ai-frontend-$BUCKET_SUFFIX/ s3://svg-ai-frontend-$BUCKET_SUFFIX/ \
  --recursive \
  --exclude "*" \
  --include "*.js" --include "*.css" \
  --metadata-directive REPLACE \
  --cache-control "public, max-age=86400"
```
- [ ] Set no-cache for HTML:
```bash
aws s3 cp s3://svg-ai-frontend-$BUCKET_SUFFIX/ s3://svg-ai-frontend-$BUCKET_SUFFIX/ \
  --recursive \
  --exclude "*" \
  --include "*.html" \
  --metadata-directive REPLACE \
  --cache-control "no-cache"
```
- [ ] Get S3 website URL:
```bash
echo "Frontend URL: http://svg-ai-frontend-$BUCKET_SUFFIX.s3-website-us-east-1.amazonaws.com"
```
- [ ] Test S3 website in browser
- [ ] Test file upload functionality
- [ ] Test AI conversion

### 3.3 Setup CloudFront (10 mins)
- [ ] Create CloudFront distribution:
```bash
aws cloudfront create-distribution \
  --origin-domain-name svg-ai-frontend-$BUCKET_SUFFIX.s3-website-us-east-1.amazonaws.com \
  --default-root-object index.html \
  --output json | tee cloudfront.json
```
- [ ] Get distribution ID:
```bash
export CF_DIST_ID=$(cat cloudfront.json | grep '"Id"' | head -1 | cut -d'"' -f4)
echo "CloudFront Distribution ID: $CF_DIST_ID"
```
- [ ] Get CloudFront domain:
```bash
export CF_DOMAIN=$(aws cloudfront get-distribution --id $CF_DIST_ID \
  --query 'Distribution.DomainName' --output text)
echo "CloudFront URL: https://$CF_DOMAIN"
```
- [ ] Wait for deployment (15-20 mins):
```bash
aws cloudfront wait distribution-deployed --id $CF_DIST_ID
```
- [ ] Test CloudFront URL in browser
- [ ] Document URLs:
```bash
echo "S3_WEBSITE=http://svg-ai-frontend-$BUCKET_SUFFIX.s3-website-us-east-1.amazonaws.com" >> .env.production
echo "CLOUDFRONT_URL=https://$CF_DOMAIN" >> .env.production
```

## 🔧 Phase 4: Optimization & Security (30 mins)

### 4.1 Enable HTTPS with Let's Encrypt (15 mins)
- [ ] SSH to server: `ssh svg-ai`
- [ ] Install Certbot:
```bash
sudo apt install -y certbot python3-certbot-nginx
```
- [ ] Create Elastic IP (for stable domain):
```bash
# On local machine
aws ec2 allocate-address --domain vpc --output json | tee elastic-ip.json
export ELASTIC_IP=$(cat elastic-ip.json | grep PublicIp | cut -d'"' -f4)
```
- [ ] Associate Elastic IP:
```bash
aws ec2 associate-address \
  --instance-id $INSTANCE_ID \
  --public-ip $ELASTIC_IP
```
- [ ] Update EC2_IP variable: `export EC2_IP=$ELASTIC_IP`
- [ ] If you have a domain, point it to Elastic IP
- [ ] For testing, use nip.io:
```bash
export DOMAIN="svg-ai-$ELASTIC_IP.nip.io"
echo "Your domain: $DOMAIN"
```
- [ ] Update Nginx with domain:
```bash
ssh svg-ai
sudo sed -i "s/server_name _;/server_name $DOMAIN;/" /etc/nginx/sites-available/svg-ai
sudo nginx -t && sudo systemctl reload nginx
```
- [ ] Get SSL certificate:
```bash
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m andy@example.com
```

### 4.2 Performance Optimization (10 mins)
- [ ] Enable gzip in Nginx:
```bash
sudo tee -a /etc/nginx/sites-available/svg-ai << 'EOF'

    # Gzip Settings
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml application/atom+xml image/svg+xml text/javascript application/vnd.ms-fontobject application/x-font-ttf font/opentype;
EOF
```
- [ ] Add response caching:
```bash
sudo tee -a /etc/nginx/sites-available/svg-ai << 'EOF'

    # Cache static responses
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
EOF
```
- [ ] Restart Nginx: `sudo systemctl restart nginx`
- [ ] Install monitoring:
```bash
sudo apt install -y htop nethogs iotop
pip install glances
```
- [ ] Create swap file (prevent OOM):
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 4.3 Monitoring & Logging (5 mins)
- [ ] Setup CloudWatch agent:
```bash
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb
```
- [ ] Configure basic monitoring:
```bash
sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "metrics": {
    "namespace": "SVG-AI",
    "metrics_collected": {
      "mem": {
        "measurement": [{"name": "mem_used_percent"}]
      },
      "disk": {
        "measurement": [{"name": "used_percent"}],
        "resources": ["/"]
      }
    }
  }
}
EOF
```
- [ ] Start CloudWatch agent:
```bash
sudo systemctl start amazon-cloudwatch-agent
sudo systemctl enable amazon-cloudwatch-agent
```
- [ ] Create log rotation:
```bash
sudo tee /etc/logrotate.d/svg-ai << 'EOF'
/var/log/svg-ai/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0640 svgai svgai
    sharedscripts
    postrotate
        systemctl reload svg-ai
    endscript
}
EOF
```

## 🚀 Phase 5: Production Launch (20 mins)

### 5.1 Testing Checklist (10 mins)
- [ ] Test health endpoint: `curl https://$DOMAIN/health`
- [ ] Test file upload via API:
```bash
curl -X POST https://$DOMAIN/api/upload \
  -F "file=@data/raw_logos/103944.png"
```
- [ ] Test conversion:
```bash
curl -X POST https://$DOMAIN/api/convert \
  -H "Content-Type: application/json" \
  -d '{"file_id":"test","converter_type":"vtracer"}'
```
- [ ] Load test with Apache Bench:
```bash
ab -n 100 -c 10 https://$DOMAIN/health
```
- [ ] Check memory usage: `ssh svg-ai "free -h"`
- [ ] Check disk usage: `ssh svg-ai "df -h"`
- [ ] Check service status: `ssh svg-ai "sudo systemctl status svg-ai"`
- [ ] Check Nginx status: `ssh svg-ai "sudo systemctl status nginx"`
- [ ] Check logs: `ssh svg-ai "sudo journalctl -u svg-ai --since '1 hour ago'"`

### 5.2 Backup & Recovery Setup (5 mins)
- [ ] Create AMI backup:
```bash
aws ec2 create-image \
  --instance-id $INSTANCE_ID \
  --name "svg-ai-backup-$(date +%Y%m%d)" \
  --description "SVG-AI Production Backup"
```
- [ ] Create startup script backup:
```bash
ssh svg-ai "sudo tar -czf /tmp/svg-ai-config.tar.gz /etc/nginx/sites-available/svg-ai /etc/systemd/system/svg-ai.service /opt/svg-ai/.env"
scp svg-ai:/tmp/svg-ai-config.tar.gz ./backups/
```
- [ ] Document recovery process:
```bash
cat > RECOVERY.md << 'EOF'
# Recovery Process
1. Launch new t3.micro from AMI
2. Associate Elastic IP
3. Restore config from backup
4. Start services
EOF
```

### 5.3 Cost Optimization (5 mins)
- [ ] Set up billing alarm:
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name svg-ai-billing \
  --alarm-description "Alert when costs exceed $20" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 20 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD
```
- [ ] Schedule instance stop/start (if not 24/7):
```bash
# Stop at 11 PM EST daily
aws events put-rule \
  --name stop-svg-ai \
  --schedule-expression "cron(0 4 * * ? *)"

# Start at 7 AM EST daily
aws events put-rule \
  --name start-svg-ai \
  --schedule-expression "cron(0 12 * * ? *)"
```
- [ ] Enable S3 lifecycle for old uploads:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket svg-ai-uploads-$BUCKET_SUFFIX \
  --lifecycle-configuration file://s3-lifecycle.json
```

## 📊 Phase 6: Monitoring Dashboard (15 mins)

### 6.1 Create Simple Monitoring Page (10 mins)
- [ ] Create monitoring script on server:
```bash
ssh svg-ai
sudo tee /opt/svg-ai/monitor.py << 'EOF'
#!/usr/bin/env python3
import psutil
import json
import datetime

def get_stats():
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "boot_time": datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
    }

if __name__ == "__main__":
    print(json.dumps(get_stats()))
EOF
```
- [ ] Add monitor endpoint to Nginx:
```bash
sudo tee -a /etc/nginx/sites-available/svg-ai << 'EOF'

    location /monitor {
        add_header Content-Type application/json;
        return 200 '{"status":"ok","server":"svg-ai"}';
    }
EOF
```
- [ ] Reload Nginx: `sudo nginx -s reload`
- [ ] Test monitor endpoint: `curl https://$DOMAIN/monitor`

### 6.2 Create Status Page (5 mins)
- [ ] Create simple status page:
```html
cat > frontend/status.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>SVG-AI Status</title>
    <meta http-equiv="refresh" content="30">
</head>
<body>
    <h1>SVG-AI System Status</h1>
    <div id="status">Checking...</div>
    <script>
        fetch('/health')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status').innerHTML =
                    `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            })
            .catch(e => {
                document.getElementById('status').innerHTML = 'Error: ' + e;
            });
    </script>
</body>
</html>
EOF
```
- [ ] Upload to S3:
```bash
aws s3 cp frontend/status.html s3://svg-ai-frontend-$BUCKET_SUFFIX/
```
- [ ] Test status page

## ✅ Deployment Complete!

### Access Points
- **API Endpoint**: `https://$DOMAIN/api/`
- **Health Check**: `https://$DOMAIN/health`
- **Frontend**: `https://$CF_DOMAIN`
- **Status Page**: `https://$CF_DOMAIN/status.html`

### Quick Commands
```bash
# SSH to server
ssh svg-ai

# Restart application
ssh svg-ai "sudo systemctl restart svg-ai"

# View logs
ssh svg-ai "sudo journalctl -u svg-ai -f"

# Check status
ssh svg-ai "sudo systemctl status svg-ai nginx"

# Update code
scp svg-ai-deploy.tar.gz svg-ai:/tmp/
ssh svg-ai "cd /opt/svg-ai && tar -xzf /tmp/svg-ai-deploy.tar.gz && sudo systemctl restart svg-ai"
```

### Monthly Costs
- **EC2 t3.micro**: $0 (free tier) or $8
- **Elastic IP**: $3.60 (if instance stopped)
- **S3 Storage**: ~$1
- **CloudFront**: ~$1
- **Total**: ~$5-13/month

### Scaling Options
1. **Vertical**: Upgrade to t3.small ($16/month)
2. **Horizontal**: Add Load Balancer + Auto Scaling ($30/month)
3. **Serverless**: Migrate to Lambda (pay per request)

## 🎉 Congratulations!
You've deployed SVG-AI for 85% less than ECS while maintaining full control!

### Next Steps
- [ ] Add custom domain
- [ ] Enable CloudWatch detailed monitoring
- [ ] Set up automated backups
- [ ] Add Redis for caching
- [ ] Implement CI/CD with GitHub Actions