# 🆓 AWS Free Tier Deployment Plan for SVG-AI

## 💰 Free Tier Benefits (12 Months)
- **EC2**: 750 hours/month t2.micro (24/7 for entire month)
- **S3**: 5 GB storage + 20K GET + 2K PUT requests/month
- **CloudFront**: 50 GB transfer + 2M requests/month
- **EBS**: 30 GB General Purpose SSD storage
- **Data Transfer**: 15 GB/month outbound
- **Total Cost**: **$0/month for first year** ✨

## 🎯 Architecture for Free Tier
```
Internet → CloudFront (Free: 50GB/month)
             ↓
          S3 Website (Free: 5GB storage)
             ↓
         t2.micro EC2 (Free: 750 hours/month)
             ↓
          S3 Models (Free: under 5GB total)
```

## 🔑 AWS Account Details
- **Account ID**: 300079938592
- **IAM User**: Andy
- **Region**: us-east-1 (best free tier coverage)
- **Free Tier Dashboard**: https://console.aws.amazon.com/billing/home#/freetier

## 📋 Phase 0: Free Tier Verification (10 mins)

### Check Free Tier Eligibility
- [x] Login to AWS Console: https://console.aws.amazon.com
- [x] Go to Billing → Free Tier: https://console.aws.amazon.com/billing/home#/freetier
- [x] Verify account is < 12 months old (shows free tier eligibility)
- [x] Check current usage:
  - [x] EC2 hours used this month
  - [x] S3 storage used
  - [x] Data transfer used
- [x] Set up billing alerts:
```bash
aws budgets create-budget \
  --account-id 300079938592 \
  --budget '{
    "BudgetName": "FreeTierMonitor",
    "BudgetLimit": {
      "Amount": "1",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```
- [ ] Create $5 billing alarm (safety net):
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "BillingAlarm" \
  --alarm-description "Alert if bill exceeds $5" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD \
  --alarm-actions arn:aws:sns:us-east-1:300079938592:billing-alerts
```

### Prepare Local Environment
- [ ] Verify git repo clean: `git status`
- [ ] Create free-tier branch: `git checkout -b free-tier-deploy`
- [ ] Check model sizes (must be under 5GB total):
```bash
du -sh models/production/
# Should show ~7MB total - well under 5GB limit
```
- [ ] Test local backend: `python -m backend.app`
- [ ] Test AI endpoints: `python test_api_convert.py`

## 🏗️ Phase 1: Minimal Infrastructure (30 mins)

### 1.1 Create Security Group (5 mins)
- [ ] Create security group:
```bash
aws ec2 create-security-group \
  --group-name svg-ai-free-sg \
  --description "SVG-AI Free Tier Security Group" \
  --output json > sg-free.json
```
- [ ] Get SG ID: `export SG_ID=$(jq -r '.GroupId' sg-free.json)`
- [ ] Add SSH (your IP only):
```bash
export MY_IP=$(curl -s https://api.ipify.org)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 22 --cidr $MY_IP/32
```
- [ ] Add HTTP/HTTPS:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 443 --cidr 0.0.0.0/0
```

### 1.2 Create Key Pair (2 mins)
- [ ] Create key pair:
```bash
aws ec2 create-key-pair \
  --key-name svg-ai-free-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/svg-ai-free.pem
chmod 400 ~/.ssh/svg-ai-free.pem
```

### 1.3 Launch t2.micro Instance (5 mins)
- [ ] Get Ubuntu 22.04 AMI:
```bash
export AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query 'Images[0].ImageId' --output text)
```
- [ ] Launch instance (FREE TIER):
```bash
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t2.micro \
  --key-name svg-ai-free-key \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=svg-ai-free}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp2}' \
  --output json > instance-free.json
```
- [ ] Get instance details:
```bash
export INSTANCE_ID=$(jq -r '.Instances[0].InstanceId' instance-free.json)
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
export EC2_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance: $INSTANCE_ID, IP: $EC2_IP"
```

### 1.4 Create S3 Buckets (FREE TIER) (8 mins)
- [ ] Create unique suffix: `export SUFFIX=$(date +%s)`
- [ ] Create frontend bucket:
```bash
aws s3 mb s3://svg-ai-free-frontend-$SUFFIX
export FRONTEND_BUCKET=svg-ai-free-frontend-$SUFFIX
```
- [ ] Create models bucket:
```bash
aws s3 mb s3://svg-ai-free-models-$SUFFIX
export MODELS_BUCKET=svg-ai-free-models-$SUFFIX
```
- [ ] Enable static website hosting:
```bash
aws s3 website s3://$FRONTEND_BUCKET \
  --index-document index.html \
  --error-document error.html
```
- [ ] Create public read policy:
```bash
cat > bucket-policy-free.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::$FRONTEND_BUCKET/*"
  }]
}
EOF
aws s3api put-bucket-policy --bucket $FRONTEND_BUCKET --policy file://bucket-policy-free.json
```
- [ ] Upload models (verify under 5GB limit):
```bash
aws s3 sync models/production/ s3://$MODELS_BUCKET/ --storage-class STANDARD
aws s3 ls s3://$MODELS_BUCKET/ --recursive --human-readable --summarize
```
- [ ] Save environment:
```bash
cat > .env.freetier << EOF
INSTANCE_ID=$INSTANCE_ID
EC2_IP=$EC2_IP
SG_ID=$SG_ID
FRONTEND_BUCKET=$FRONTEND_BUCKET
MODELS_BUCKET=$MODELS_BUCKET
SUFFIX=$SUFFIX
EOF
```

### 1.5 Setup SSH Config (2 mins)
- [ ] Add to SSH config:
```bash
cat >> ~/.ssh/config << EOF

Host svg-ai-free
  HostName $EC2_IP
  User ubuntu
  IdentityFile ~/.ssh/svg-ai-free.pem
  ServerAliveInterval 60
EOF
```
- [ ] Test connection: `ssh svg-ai-free "echo 'Connected to free tier instance!'"`

## 🖥️ Phase 2: Minimal Server Setup (30 mins)

### 2.1 Basic Server Configuration (10 mins)
- [ ] SSH to server: `ssh svg-ai-free`
- [ ] Update system (efficient updates only):
```bash
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3-pip nginx git
sudo apt install -y build-essential python3.9-dev
```
- [ ] Create minimal app structure:
```bash
sudo mkdir -p /opt/svg-ai
sudo chown ubuntu:ubuntu /opt/svg-ai
cd /opt/svg-ai
python3.9 -m venv venv
source venv/bin/activate
```

### 2.2 Deploy Application (Minimal) (15 mins)
- [ ] Create deployment archive (minimal size):
```bash
# On local machine
tar -czf svg-ai-minimal.tar.gz \
  --exclude='.git*' \
  --exclude='venv*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data/logos' \
  --exclude='uploads' \
  --exclude='reports' \
  --exclude='htmlcov' \
  --exclude='benchmark_test' \
  backend frontend models/production requirements.txt
```
- [ ] Copy to server:
```bash
scp svg-ai-minimal.tar.gz svg-ai-free:/tmp/
```
- [ ] Extract and setup:
```bash
ssh svg-ai-free
cd /opt/svg-ai
tar -xzf /tmp/svg-ai-minimal.tar.gz
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
export TMPDIR=/tmp
pip install --no-cache-dir vtracer
```
- [ ] Create minimal environment:
```bash
cat > .env << 'EOF'
FLASK_ENV=production
MODEL_DIR=/opt/svg-ai/models/production
UPLOAD_DIR=/tmp/uploads
CACHE_DIR=/tmp/cache
LOG_LEVEL=WARNING
MAX_FILE_SIZE_MB=5
EOF
```

### 2.3 Create Systemd Service (5 mins)
- [ ] Create service file:
```bash
sudo tee /etc/systemd/system/svg-ai-free.service << 'EOF'
[Unit]
Description=SVG-AI Free Tier Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/svg-ai
Environment="PATH=/opt/svg-ai/venv/bin"
ExecStart=/opt/svg-ai/venv/bin/python -m backend.app
Restart=always
RestartSec=10
MemoryLimit=800M

[Install]
WantedBy=multi-user.target
EOF
```
- [ ] Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable svg-ai-free
sudo systemctl start svg-ai-free
sudo systemctl status svg-ai-free
```
- [ ] Test locally: `curl http://localhost:8001/health`

## 🌐 Phase 3: Nginx + SSL (FREE) (20 mins)

### 3.1 Configure Nginx (10 mins)
- [ ] Create Nginx config:
```bash
sudo tee /etc/nginx/sites-available/svg-ai-free << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 10M;

    # Minimize logging for free tier
    access_log off;
    error_log /var/log/nginx/error.log warn;

    location /api/ {
        proxy_pass http://127.0.0.1:8001/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 60;
        proxy_connect_timeout 10;
    }

    location /health {
        proxy_pass http://127.0.0.1:8001/health;
    }

    location / {
        return 200 'SVG-AI Free Tier Server - API endpoint: /api/\n';
        add_header Content-Type text/plain;
    }
}
EOF
```
- [ ] Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/svg-ai-free /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```
- [ ] Test from local: `curl http://$EC2_IP/health`

### 3.2 Setup Free SSL with Let's Encrypt (10 mins)
- [ ] Install Certbot:
```bash
sudo apt install -y certbot python3-certbot-nginx
```
- [ ] Get free domain using nip.io:
```bash
export DOMAIN="svg-ai-$EC2_IP.nip.io"
echo "Your free domain: $DOMAIN"
```
- [ ] Update Nginx with domain:
```bash
sudo sed -i "s/server_name _;/server_name $DOMAIN;/" /etc/nginx/sites-available/svg-ai-free
sudo nginx -t && sudo systemctl reload nginx
```
- [ ] Get SSL certificate (FREE):
```bash
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email andy@example.com
```
- [ ] Test HTTPS: `curl https://$DOMAIN/health`

## 🎨 Phase 4: Frontend Deployment (FREE TIER) (15 mins)

### 4.1 Prepare Frontend for Free Tier (5 mins)
- [ ] Update API endpoint (local machine):
```bash
sed -i "" "s|localhost:8001|$DOMAIN|g" frontend/js/modules/converter.js
sed -i "" "s|http://|https://|g" frontend/js/modules/converter.js
```
- [ ] Test frontend locally: `python -m http.server 8080`

### 4.2 Deploy to S3 (FREE) (5 mins)
- [ ] Upload to S3 (staying under free tier limits):
```bash
aws s3 sync frontend/ s3://$FRONTEND_BUCKET/ \
  --exclude "*.git*" --exclude "*.md" --exclude ".DS_Store" \
  --cache-control "public, max-age=86400"
```
- [ ] Set HTML no-cache:
```bash
aws s3 cp s3://$FRONTEND_BUCKET/ s3://$FRONTEND_BUCKET/ \
  --recursive --exclude "*" --include "*.html" \
  --metadata-directive REPLACE --cache-control "no-cache"
```
- [ ] Get S3 website URL:
```bash
export S3_URL="http://$FRONTEND_BUCKET.s3-website-us-east-1.amazonaws.com"
echo "S3 Website: $S3_URL"
```

### 4.3 Setup CloudFront (FREE TIER) (5 mins)
- [ ] Create minimal CloudFront distribution:
```bash
aws cloudfront create-distribution \
  --origin-domain-name $FRONTEND_BUCKET.s3-website-us-east-1.amazonaws.com \
  --default-root-object index.html \
  --comment "SVG-AI Free Tier" \
  --output json > cloudfront-free.json
```
- [ ] Get CloudFront details:
```bash
export CF_ID=$(jq -r '.Distribution.Id' cloudfront-free.json)
export CF_URL=$(jq -r '.Distribution.DomainName' cloudfront-free.json)
echo "CloudFront URL: https://$CF_URL (deploying...)"
```
- [ ] Wait for deployment (15-20 mins):
```bash
echo "Waiting for CloudFront deployment..."
aws cloudfront wait distribution-deployed --id $CF_ID
echo "CloudFront deployed!"
```

## 📊 Phase 5: Free Tier Monitoring (15 mins)

### 5.1 Setup Usage Monitoring (10 mins)
- [ ] Create usage check script:
```bash
ssh svg-ai-free
sudo tee /opt/svg-ai/check-usage.sh << 'EOF'
#!/bin/bash
echo "=== FREE TIER USAGE CHECK ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo "Memory: $(free -h)"
echo "Disk: $(df -h /)"
echo "Network: $(cat /proc/net/dev | grep eth0)"
echo "Processes: $(ps aux | grep svg-ai | grep -v grep)"
EOF
chmod +x /opt/svg-ai/check-usage.sh
```
- [ ] Add to crontab (daily check):
```bash
echo "0 8 * * * /opt/svg-ai/check-usage.sh >> /tmp/daily-usage.log 2>&1" | crontab -
```
- [ ] Create simple monitoring endpoint:
```bash
sudo tee -a /etc/nginx/sites-available/svg-ai-free << 'EOF'

    location /usage {
        alias /tmp/daily-usage.log;
        add_header Content-Type text/plain;
    }
EOF
sudo nginx -s reload
```

### 5.2 Create Cost Dashboard (5 mins)
- [ ] Create simple status page:
```bash
cat > frontend/free-tier-status.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>SVG-AI Free Tier Status</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .ok { background: #d4edda; border: 1px solid #c3e6cb; }
        .warning { background: #fff3cd; border: 1px solid #ffeaa7; }
    </style>
</head>
<body>
    <h1>🆓 SVG-AI Free Tier Status</h1>

    <div class="status ok">
        <h3>Service Status</h3>
        <p>✅ API: <span id="api-status">Checking...</span></p>
        <p>✅ AI Models: <span id="ai-status">Checking...</span></p>
    </div>

    <div class="status warning">
        <h3>Free Tier Limits</h3>
        <p>⚠️ EC2: 750 hours/month (24/7 = 720 hours)</p>
        <p>⚠️ S3: 5 GB storage limit</p>
        <p>⚠️ Data Transfer: 15 GB/month outbound</p>
    </div>

    <div class="status ok">
        <h3>Quick Test</h3>
        <button onclick="testAPI()">Test Conversion</button>
        <div id="test-result"></div>
    </div>

    <script>
        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('api-status').innerHTML = '🟢 Online';
                document.getElementById('ai-status').innerHTML = data.ai_enabled ? '🟢 Available' : '🟡 Basic Mode';
            } catch (e) {
                document.getElementById('api-status').innerHTML = '🔴 Offline';
            }
        }

        async function testAPI() {
            document.getElementById('test-result').innerHTML = 'Testing...';
            try {
                const response = await fetch('/api/convert', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({converter_type: 'vtracer'})
                });
                document.getElementById('test-result').innerHTML =
                    response.ok ? '✅ API Working' : '❌ API Error';
            } catch (e) {
                document.getElementById('test-result').innerHTML = '❌ Connection Failed';
            }
        }

        checkStatus();
    </script>
</body>
</html>
EOF
```
- [ ] Upload status page:
```bash
aws s3 cp frontend/free-tier-status.html s3://$FRONTEND_BUCKET/
```

## 📝 Phase 6: Documentation & Maintenance (10 mins)

### 6.1 Create Usage Guidelines (5 mins)
- [ ] Document free tier limits:
```bash
cat > FREE_TIER_USAGE.md << 'EOF'
# Free Tier Usage Guidelines

## Monthly Limits (Don't Exceed!)
- **EC2**: 750 hours (run 24/7 = 720 hours, 30 hours buffer)
- **S3 Storage**: 5 GB total (currently using ~7MB)
- **S3 Requests**: 20K GET, 2K PUT per month
- **Data Transfer**: 15 GB outbound per month
- **CloudFront**: 50 GB transfer, 2M requests per month

## Current Usage Check
```bash
# Check EC2 hours this month
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 --granularity MONTHLY --metrics UsageQuantity --group-by Type=DIMENSION,Key=SERVICE | grep EC2

# Check S3 storage
aws s3 ls s3://svg-ai-free-frontend-$SUFFIX --recursive --summarize
aws s3 ls s3://svg-ai-free-models-$SUFFIX --recursive --summarize
```

## Emergency Stop (If Approaching Limits)
```bash
# Stop instance to save hours
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start when needed
aws ec2 start-instances --instance-ids $INSTANCE_ID
```

## After 12 Months
- t2.micro: $8.50/month
- S3: ~$0.10/month
- CloudFront: ~$1/month
- Total: ~$10/month
EOF
```

### 6.2 Create Backup Script (5 mins)
- [ ] Create minimal backup:
```bash
ssh svg-ai-free
sudo tee /opt/svg-ai/backup.sh << 'EOF'
#!/bin/bash
# Minimal backup script for free tier
DATE=$(date +%Y%m%d)
tar -czf /tmp/svg-ai-backup-$DATE.tar.gz \
  /opt/svg-ai/.env \
  /etc/nginx/sites-available/svg-ai-free \
  /etc/systemd/system/svg-ai-free.service
echo "Backup created: /tmp/svg-ai-backup-$DATE.tar.gz"
EOF
chmod +x /opt/svg-ai/backup.sh
```
- [ ] Run initial backup: `/opt/svg-ai/backup.sh`

## 🎯 Final Setup Summary

### Access Points
- **API Endpoint**: `https://$DOMAIN/api/`
- **Health Check**: `https://$DOMAIN/health`
- **Frontend**: `https://$CF_URL`
- **Status Page**: `https://$CF_URL/free-tier-status.html`
- **SSH Access**: `ssh svg-ai-free`

### Free Tier Status
```bash
# Load environment
source .env.freetier

# Check everything is working
curl https://$DOMAIN/health
curl https://$CF_URL

# Monitor free tier usage
aws ce get-cost-and-usage --time-period Start=$(date -d 'first day of this month' +%Y-%m-%d),End=$(date -d 'first day of next month' +%Y-%m-%d) --granularity MONTHLY --metrics BlendedCost --group-by Type=DIMENSION,Key=SERVICE
```

### Daily Maintenance
```bash
# Check service status
ssh svg-ai-free "sudo systemctl status svg-ai-free nginx"

# Check usage
ssh svg-ai-free "cat /tmp/daily-usage.log | tail -20"

# Check free tier dashboard
# https://console.aws.amazon.com/billing/home#/freetier
```

## 🚨 Free Tier Warnings

### Critical Limits
- ⚠️ **EC2 Hours**: Stop instance before 750 hours/month
- ⚠️ **Data Transfer**: Monitor outbound traffic
- ⚠️ **S3 Requests**: Don't exceed 20K GET requests/month

### Cost Triggers
- Elastic IP when instance stopped: $3.60/month
- EBS snapshots: $0.05/GB/month
- CloudWatch detailed monitoring: $3.50/month

### Monitoring Commands
```bash
# Check monthly costs
aws ce get-cost-and-usage --time-period Start=$(date -d 'first day of this month' +%Y-%m-%d),End=$(date +%Y-%m-%d) --granularity MONTHLY --metrics BlendedCost

# Check free tier usage
aws support describe-service-levels # Free tier status
```

## ✅ Deployment Complete - $0/month for 12 months!

### What You Get for FREE:
- ✅ 24/7 SVG conversion API
- ✅ AI-enhanced conversions
- ✅ HTTPS with SSL certificate
- ✅ CDN with CloudFront
- ✅ Professional architecture
- ✅ Monitoring and backups

### Next Steps After Free Tier (Month 13):
- Continue on t2.micro: ~$10/month
- Upgrade to t3.small: ~$16/month
- Add Redis caching: +$13/month
- Custom domain: +$12/year

**Congratulations! You've deployed a production-grade application for $0/month! 🎉**