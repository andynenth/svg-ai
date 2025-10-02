# 🚀 AWS Production Deployment Checklist for SVG-AI

## 📋 Pre-Deployment Preparation

### Account & Access Setup (2 hours)
- [ ] Create AWS account or use existing
- [ ] Install AWS CLI: `brew install awscli`
- [ ] Configure AWS credentials: `aws configure`
- [ ] Create IAM user for deployment with programmatic access
- [ ] Attach `AdministratorAccess` policy temporarily (restrict later)
- [ ] Install Terraform: `brew install terraform`
- [ ] Install Docker Desktop for Mac
- [ ] Create deployment workspace: `mkdir ~/svg-ai-deploy`

### Local Testing (1 hour)
- [ ] Run full test suite: `pytest tests/`
- [ ] Test AI endpoints locally: `python test_api_convert.py`
- [ ] Verify frontend works: `http://localhost:8080/frontend/`
- [ ] Check model files exist: `ls -la models/production/`
- [ ] Document current API endpoints in `API_ENDPOINTS.md`
- [ ] Create `.env.production` file with production configs

## 🏗️ Phase 1: Core Infrastructure (Day 1)

### 1.1 S3 Buckets Creation (30 mins)
- [ ] Create main bucket: `aws s3 mb s3://svg-ai-production-{random-id}`
- [ ] Create frontend bucket: `aws s3 mb s3://svg-ai-frontend-{random-id}`
- [ ] Create models bucket: `aws s3 mb s3://svg-ai-models-{random-id}`
- [ ] Create uploads bucket: `aws s3 mb s3://svg-ai-uploads-{random-id}`
- [ ] Enable versioning: `aws s3api put-bucket-versioning --bucket svg-ai-production-{id} --versioning-configuration Status=Enabled`
- [ ] Configure CORS for frontend bucket
- [ ] Set bucket policies for public read on frontend bucket
- [ ] Create lifecycle rules for old uploads (delete after 7 days)

### 1.2 Upload AI Models (15 mins)
- [ ] Compress models: `tar -czf models.tar.gz models/production/`
- [ ] Upload to S3: `aws s3 cp models.tar.gz s3://svg-ai-models-{id}/`
- [ ] Upload individual models:
  - [ ] `aws s3 cp models/production/logo_classifier.torchscript s3://svg-ai-models-{id}/`
  - [ ] `aws s3 cp models/production/quality_predictor.torchscript s3://svg-ai-models-{id}/`
  - [ ] `aws s3 cp models/production/correlation_models.pkl s3://svg-ai-models-{id}/`
- [ ] Set public read permissions for model files
- [ ] Test download: `aws s3 cp s3://svg-ai-models-{id}/logo_classifier.torchscript /tmp/test.model`

### 1.3 VPC Setup (45 mins)
- [ ] Create VPC: `aws ec2 create-vpc --cidr-block 10.0.0.0/16`
- [ ] Create Internet Gateway: `aws ec2 create-internet-gateway`
- [ ] Attach IGW to VPC: `aws ec2 attach-internet-gateway --vpc-id {vpc-id} --internet-gateway-id {igw-id}`
- [ ] Create public subnet AZ1: `aws ec2 create-subnet --vpc-id {vpc-id} --cidr-block 10.0.1.0/24 --availability-zone us-east-1a`
- [ ] Create public subnet AZ2: `aws ec2 create-subnet --vpc-id {vpc-id} --cidr-block 10.0.2.0/24 --availability-zone us-east-1b`
- [ ] Create private subnet AZ1: `aws ec2 create-subnet --vpc-id {vpc-id} --cidr-block 10.0.10.0/24 --availability-zone us-east-1a`
- [ ] Create private subnet AZ2: `aws ec2 create-subnet --vpc-id {vpc-id} --cidr-block 10.0.20.0/24 --availability-zone us-east-1b`
- [ ] Create NAT Gateway in public subnet
- [ ] Configure route tables for public subnets (0.0.0.0/0 → IGW)
- [ ] Configure route tables for private subnets (0.0.0.0/0 → NAT)

### 1.4 Security Groups (30 mins)
- [ ] Create ALB security group:
  - [ ] Inbound: 80 from 0.0.0.0/0
  - [ ] Inbound: 443 from 0.0.0.0/0
  - [ ] Outbound: All traffic
- [ ] Create ECS security group:
  - [ ] Inbound: 8001 from ALB SG
  - [ ] Outbound: All traffic (for S3, etc.)
- [ ] Create RDS security group (if using):
  - [ ] Inbound: 5432 from ECS SG
  - [ ] Outbound: None needed
- [ ] Document security group IDs in `INFRASTRUCTURE.md`

## 🐳 Phase 2: Containerization (Day 2)

### 2.1 Create Dockerfile (1 hour)
- [ ] Create `Dockerfile` in project root
- [ ] Create `.dockerignore` file:
  ```
  *.pyc
  __pycache__
  .git
  .env
  venv/
  data/
  ```
- [ ] Write multi-stage Dockerfile:
  ```dockerfile
  FROM python:3.9-slim AS builder
  WORKDIR /app
  COPY requirements*.txt ./
  RUN pip install --no-cache-dir -r requirements.txt -r requirements_ai_phase1.txt

  FROM python:3.9-slim
  WORKDIR /app
  COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
  COPY backend/ ./backend/
  COPY models/production/ ./models/production/
  ```
- [ ] Test Docker build: `docker build -t svg-ai:test .`
- [ ] Test Docker run: `docker run -p 8001:8001 svg-ai:test`
- [ ] Verify API works: `curl http://localhost:8001/health`

### 2.2 Create ECR Repository (15 mins)
- [ ] Create ECR repo: `aws ecr create-repository --repository-name svg-ai`
- [ ] Get login token: `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin {account-id}.dkr.ecr.us-east-1.amazonaws.com`
- [ ] Tag image: `docker tag svg-ai:test {account-id}.dkr.ecr.us-east-1.amazonaws.com/svg-ai:latest`
- [ ] Push image: `docker push {account-id}.dkr.ecr.us-east-1.amazonaws.com/svg-ai:latest`
- [ ] Verify image in ECR console
- [ ] Set lifecycle policy to keep only last 10 images

### 2.3 Optimize Container (30 mins)
- [ ] Add health check to Dockerfile:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"
  ```
- [ ] Add non-root user:
  ```dockerfile
  RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
  USER appuser
  ```
- [ ] Set environment variables:
  ```dockerfile
  ENV MODEL_DIR=/app/models/production
  ENV PYTHONUNBUFFERED=1
  ```
- [ ] Minimize image size (target: < 1GB)
- [ ] Run security scan: `docker scout cves svg-ai:test`
- [ ] Document final image size

## 🚀 Phase 3: ECS Deployment (Day 3)

### 3.1 Create ECS Cluster (20 mins)
- [ ] Create Fargate cluster: `aws ecs create-cluster --cluster-name svg-ai-cluster --capacity-providers FARGATE`
- [ ] Create CloudWatch log group: `aws logs create-log-group --log-group-name /ecs/svg-ai`
- [ ] Set log retention: `aws logs put-retention-policy --log-group-name /ecs/svg-ai --retention-in-days 7`
- [ ] Create IAM role for task execution:
  ```bash
  aws iam create-role --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file://task-execution-role.json
  ```
- [ ] Attach policies to execution role:
  - [ ] `aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy`
- [ ] Create IAM role for task (application):
  ```bash
  aws iam create-role --role-name svg-ai-task-role \
    --assume-role-policy-document file://task-role.json
  ```

### 3.2 Create Task Definition (30 mins)
- [ ] Create `task-definition.json`:
  ```json
  {
    "family": "svg-ai",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::{account-id}:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::{account-id}:role/svg-ai-task-role"
  }
  ```
- [ ] Add container definition:
  - [ ] Image URI from ECR
  - [ ] Port mappings: 8001
  - [ ] Environment variables for S3 buckets
  - [ ] CloudWatch logging configuration
- [ ] Register task definition: `aws ecs register-task-definition --cli-input-json file://task-definition.json`
- [ ] Verify in ECS console
- [ ] Create revision with 4GB memory if needed for AI models

### 3.3 Create Application Load Balancer (45 mins)
- [ ] Create ALB: `aws elbv2 create-load-balancer --name svg-ai-alb --subnets {subnet-1} {subnet-2} --security-groups {alb-sg}`
- [ ] Create target group:
  ```bash
  aws elbv2 create-target-group --name svg-ai-targets \
    --protocol HTTP --port 8001 --vpc-id {vpc-id} \
    --target-type ip --health-check-path /health
  ```
- [ ] Configure health check:
  - [ ] Interval: 30 seconds
  - [ ] Timeout: 5 seconds
  - [ ] Healthy threshold: 2
  - [ ] Unhealthy threshold: 3
- [ ] Create listener:
  ```bash
  aws elbv2 create-listener --load-balancer-arn {alb-arn} \
    --protocol HTTP --port 80 \
    --default-actions Type=forward,TargetGroupArn={tg-arn}
  ```
- [ ] Test ALB DNS name in browser
- [ ] Document ALB DNS in `INFRASTRUCTURE.md`

### 3.4 Create ECS Service (30 mins)
- [ ] Create service:
  ```bash
  aws ecs create-service \
    --cluster svg-ai-cluster \
    --service-name svg-ai-service \
    --task-definition svg-ai:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[{subnet-1},{subnet-2}],securityGroups=[{ecs-sg}],assignPublicIp=ENABLED}"
  ```
- [ ] Configure with ALB:
  ```bash
  --load-balancers targetGroupArn={tg-arn},containerName=svg-ai,containerPort=8001
  ```
- [ ] Wait for service to stabilize: `aws ecs wait services-stable --cluster svg-ai-cluster --services svg-ai-service`
- [ ] Check service status: `aws ecs describe-services --cluster svg-ai-cluster --services svg-ai-service`
- [ ] Verify targets healthy in target group
- [ ] Test API via ALB: `curl http://{alb-dns}/health`

### 3.5 Configure Auto Scaling (30 mins)
- [ ] Register scalable target:
  ```bash
  aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/svg-ai-cluster/svg-ai-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 2 --max-capacity 10
  ```
- [ ] Create CPU scaling policy:
  ```bash
  aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --resource-id service/svg-ai-cluster/svg-ai-service \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-name cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration file://cpu-scaling.json
  ```
- [ ] Create memory scaling policy
- [ ] Test scaling: Generate load with Apache Bench
- [ ] Verify scale-out occurs
- [ ] Verify scale-in after load stops

## 🌐 Phase 4: Frontend Deployment (Day 4)

### 4.1 Prepare Frontend (30 mins)
- [ ] Update API endpoint in `frontend/js/modules/converter.js`:
  - [ ] Replace `localhost:8001` with ALB DNS
- [ ] Update CORS settings in backend to allow CloudFront domain
- [ ] Minify JavaScript: `terser frontend/js/*.js -o frontend/js/app.min.js`
- [ ] Minify CSS: `csso frontend/style.css -o frontend/style.min.css`
- [ ] Update HTML to use minified files
- [ ] Test locally with production API endpoint

### 4.2 Deploy to S3 (20 mins)
- [ ] Enable static website hosting:
  ```bash
  aws s3 website s3://svg-ai-frontend-{id} \
    --index-document index.html \
    --error-document error.html
  ```
- [ ] Create bucket policy for public read:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "PublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::svg-ai-frontend-{id}/*"
    }]
  }
  ```
- [ ] Upload frontend files:
  ```bash
  aws s3 sync frontend/ s3://svg-ai-frontend-{id}/ \
    --exclude ".git/*" --exclude "*.md" \
    --cache-control max-age=3600
  ```
- [ ] Test S3 website endpoint
- [ ] Verify all assets load correctly

### 4.3 Setup CloudFront (45 mins)
- [ ] Create distribution:
  ```bash
  aws cloudfront create-distribution \
    --origin-domain-name svg-ai-frontend-{id}.s3-website-us-east-1.amazonaws.com \
    --default-root-object index.html
  ```
- [ ] Configure behaviors:
  - [ ] `/api/*` → ALB origin
  - [ ] `/*` → S3 origin
- [ ] Configure caching:
  - [ ] HTML: max-age=0
  - [ ] JS/CSS: max-age=86400
  - [ ] Images: max-age=604800
- [ ] Enable compression
- [ ] Configure custom error pages:
  - [ ] 404 → /404.html
  - [ ] 403 → /index.html (for SPA routing)
- [ ] Wait for distribution deployment (15-20 mins)
- [ ] Test CloudFront URL
- [ ] Create invalidation: `aws cloudfront create-invalidation --distribution-id {dist-id} --paths "/*"`

### 4.4 Domain Setup (30 mins)
- [ ] Register domain in Route53 or external registrar
- [ ] Create hosted zone in Route53 (if using external registrar)
- [ ] Request SSL certificate in ACM:
  ```bash
  aws acm request-certificate \
    --domain-name svg-ai.com \
    --domain-name www.svg-ai.com \
    --validation-method DNS
  ```
- [ ] Add DNS validation records
- [ ] Wait for certificate validation
- [ ] Update CloudFront to use custom domain and SSL cert
- [ ] Create Route53 A record (alias) pointing to CloudFront
- [ ] Test https://svg-ai.com

## 📊 Phase 5: Monitoring & Logging (Day 5)

### 5.1 CloudWatch Dashboards (45 mins)
- [ ] Create dashboard: `aws cloudwatch put-dashboard --dashboard-name SVG-AI-Production`
- [ ] Add ECS metrics widget:
  - [ ] CPU utilization
  - [ ] Memory utilization
  - [ ] Task count
- [ ] Add ALB metrics widget:
  - [ ] Request count
  - [ ] Target response time
  - [ ] HTTP 4xx/5xx errors
- [ ] Add S3 metrics widget:
  - [ ] Bucket size
  - [ ] Number of objects
  - [ ] Request metrics
- [ ] Add CloudFront metrics widget:
  - [ ] Requests
  - [ ] Bytes downloaded
  - [ ] Error rate
- [ ] Add custom metrics for:
  - [ ] Conversion success rate
  - [ ] AI model inference time
  - [ ] Upload size distribution

### 5.2 CloudWatch Alarms (30 mins)
- [ ] ECS high CPU alarm (> 80% for 5 minutes):
  ```bash
  aws cloudwatch put-metric-alarm \
    --alarm-name svg-ai-high-cpu \
    --alarm-description "Alert when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold
  ```
- [ ] ECS high memory alarm (> 80%)
- [ ] ALB unhealthy targets alarm
- [ ] ALB 5xx errors alarm (> 1% of requests)
- [ ] ALB response time alarm (> 1 second)
- [ ] ECS task count alarm (< 1 running task)
- [ ] Create SNS topic for alarm notifications
- [ ] Subscribe email to SNS topic
- [ ] Test alarm by stopping a task

### 5.3 Application Logging (30 mins)
- [ ] Configure structured logging in Flask app
- [ ] Add request ID to all log messages
- [ ] Log conversion metrics:
  ```python
  logger.info("conversion_complete", {
    "file_id": file_id,
    "converter": converter_type,
    "duration_ms": duration,
    "quality_score": ssim,
    "file_size": size
  })
  ```
- [ ] Create CloudWatch Insights queries:
  - [ ] Conversion success rate
  - [ ] Average conversion time by type
  - [ ] Error analysis
- [ ] Set up log streaming to S3 for long-term storage
- [ ] Configure log retention policies

### 5.4 X-Ray Tracing (45 mins)
- [ ] Add X-Ray SDK to requirements.txt
- [ ] Instrument Flask app:
  ```python
  from aws_xray_sdk.core import xray_recorder
  from aws_xray_sdk.ext.flask.middleware import XRayMiddleware
  XRayMiddleware(app, xray_recorder)
  ```
- [ ] Instrument AWS SDK calls
- [ ] Instrument external HTTP calls
- [ ] Add custom segments for AI inference
- [ ] Deploy updated container
- [ ] Enable X-Ray tracing in ECS task definition
- [ ] View service map in X-Ray console
- [ ] Identify performance bottlenecks

## 🔄 Phase 6: CI/CD Pipeline (Day 6)

### 6.1 GitHub Actions Setup (45 mins)
- [ ] Create `.github/workflows/deploy.yml`
- [ ] Add AWS credentials as GitHub secrets:
  - [ ] AWS_ACCESS_KEY_ID
  - [ ] AWS_SECRET_ACCESS_KEY
  - [ ] AWS_REGION
  - [ ] ECR_REPOSITORY
  - [ ] ECS_CLUSTER
  - [ ] ECS_SERVICE
- [ ] Create build job:
  ```yaml
  - name: Build and push Docker image
    env:
      ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
      IMAGE_TAG: ${{ github.sha }}
    run: |
      docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
      docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
  ```
- [ ] Create deploy job:
  ```yaml
  - name: Update ECS service
    run: |
      aws ecs update-service \
        --cluster ${{ secrets.ECS_CLUSTER }} \
        --service ${{ secrets.ECS_SERVICE }} \
        --force-new-deployment
  ```
- [ ] Add test job before deployment
- [ ] Add manual approval for production
- [ ] Test workflow with a commit

### 6.2 Testing Pipeline (30 mins)
- [ ] Create test stage:
  ```yaml
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run tests
      run: |
        pip install -r requirements.txt
        pytest tests/ --cov=backend
  ```
- [ ] Add linting:
  ```yaml
  - name: Lint code
    run: |
      flake8 backend/
      black --check backend/
  ```
- [ ] Add security scanning:
  ```yaml
  - name: Security scan
    run: |
      pip install safety
      safety check
  ```
- [ ] Add integration tests against staging environment
- [ ] Create test report artifacts
- [ ] Require tests to pass before merge

### 6.3 Deployment Strategies (45 mins)
- [ ] Implement blue-green deployment:
  - [ ] Create two target groups (blue/green)
  - [ ] Update ALB listener rules to switch traffic
  - [ ] Add rollback capability
- [ ] Create canary deployment script:
  ```bash
  # Deploy 10% traffic to new version
  aws elbv2 modify-rule --rule-arn {rule-arn} \
    --actions Type=forward,ForwardConfig={TargetGroups=[{TargetGroupArn={blue-tg},Weight=90},{TargetGroupArn={green-tg},Weight=10}]}
  ```
- [ ] Add health check validation before full rollout
- [ ] Create rollback automation
- [ ] Document deployment procedures
- [ ] Test rollback scenario

### 6.4 Model Deployment Pipeline (30 mins)
- [ ] Create separate workflow for model updates
- [ ] Add model validation step:
  ```python
  # Validate model performance
  assert accuracy > 0.95, "Model accuracy too low"
  assert inference_time < 100, "Model too slow"
  ```
- [ ] Upload models to S3 with versioning
- [ ] Update model path in ECS environment variables
- [ ] Trigger ECS service update
- [ ] Add A/B testing capability for models
- [ ] Create model rollback procedure

## 🛡️ Phase 7: Security Hardening (Day 7)

### 7.1 IAM Least Privilege (45 mins)
- [ ] Review and restrict ECS task role:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject"
        ],
        "Resource": [
          "arn:aws:s3:::svg-ai-uploads-*/*",
          "arn:aws:s3:::svg-ai-models-*/*"
        ]
      }
    ]
  }
  ```
- [ ] Create separate IAM roles for each service
- [ ] Remove AdministratorAccess from deployment user
- [ ] Create custom deployment policy with minimum permissions
- [ ] Enable MFA for AWS console access
- [ ] Rotate access keys
- [ ] Document IAM roles and policies

### 7.2 Secrets Management (30 mins)
- [ ] Create secrets in Secrets Manager:
  ```bash
  aws secretsmanager create-secret \
    --name svg-ai/production/api-key \
    --secret-string '{"key":"your-api-key"}'
  ```
- [ ] Update ECS task definition to use secrets:
  ```json
  "secrets": [
    {
      "name": "API_KEY",
      "valueFrom": "arn:aws:secretsmanager:us-east-1:{account}:secret:svg-ai/production/api-key"
    }
  ]
  ```
- [ ] Rotate secrets: Create rotation Lambda
- [ ] Update application to read from environment variables
- [ ] Remove hardcoded credentials from code
- [ ] Test secret rotation

### 7.3 Network Security (30 mins)
- [ ] Enable VPC Flow Logs:
  ```bash
  aws ec2 create-flow-logs \
    --resource-type VPC \
    --traffic-type ALL \
    --resource-ids {vpc-id} \
    --log-destination-type s3 \
    --log-destination arn:aws:s3:::svg-ai-logs/vpc-flow-logs/
  ```
- [ ] Review and tighten security groups
- [ ] Implement AWS WAF on ALB:
  - [ ] Rate limiting rule (1000 requests per minute)
  - [ ] SQL injection protection
  - [ ] XSS protection
  - [ ] IP reputation lists
- [ ] Enable AWS Shield Standard (automatic)
- [ ] Configure network ACLs as additional layer
- [ ] Document network architecture

### 7.4 Data Security (30 mins)
- [ ] Enable S3 bucket encryption:
  ```bash
  aws s3api put-bucket-encryption \
    --bucket svg-ai-uploads-{id} \
    --server-side-encryption-configuration file://encryption.json
  ```
- [ ] Enable S3 access logging
- [ ] Configure S3 Object Lock for compliance
- [ ] Enable CloudTrail for API auditing:
  ```bash
  aws cloudtrail create-trail \
    --name svg-ai-audit \
    --s3-bucket-name svg-ai-logs
  ```
- [ ] Configure data lifecycle policies
- [ ] Implement GDPR compliance (if needed):
  - [ ] Add data deletion API
  - [ ] Create privacy policy
  - [ ] Log consent

## 💰 Phase 8: Cost Optimization (Day 8)

### 8.1 Cost Analysis (30 mins)
- [ ] Enable Cost Explorer
- [ ] Create cost allocation tags:
  - [ ] Environment: production
  - [ ] Service: svg-ai
  - [ ] Component: frontend/backend/storage
- [ ] Tag all resources appropriately
- [ ] Create monthly budget:
  ```bash
  aws budgets create-budget \
    --account-id {account} \
    --budget file://budget.json \
    --notifications-with-subscribers file://notifications.json
  ```
- [ ] Set up budget alerts at 80% and 100%
- [ ] Review last month's costs
- [ ] Identify top cost drivers

### 8.2 Compute Optimization (30 mins)
- [ ] Analyze ECS task CPU/memory usage
- [ ] Right-size container resources:
  - [ ] CPU: 512 (if usage < 30%)
  - [ ] Memory: 1024 (if usage < 50%)
- [ ] Consider Fargate Spot for non-critical tasks:
  ```json
  "capacityProviderStrategy": [
    {
      "capacityProvider": "FARGATE_SPOT",
      "weight": 80
    },
    {
      "capacityProvider": "FARGATE",
      "weight": 20
    }
  ]
  ```
- [ ] Implement request batching to reduce invocations
- [ ] Add caching layer to reduce compute needs
- [ ] Schedule scale-down during off-peak hours

### 8.3 Storage Optimization (30 mins)
- [ ] Enable S3 Intelligent-Tiering:
  ```bash
  aws s3api put-bucket-intelligent-tiering-configuration \
    --bucket svg-ai-uploads-{id} \
    --id archive-old-files \
    --intelligent-tiering-configuration file://tiering.json
  ```
- [ ] Set lifecycle rules for old conversions:
  - [ ] Standard → IA after 30 days
  - [ ] IA → Glacier after 90 days
  - [ ] Delete after 365 days
- [ ] Compress large files before storage
- [ ] Implement S3 Transfer Acceleration for uploads
- [ ] Clean up unused EBS snapshots
- [ ] Delete old CloudWatch logs

### 8.4 Data Transfer Optimization (20 mins)
- [ ] Enable CloudFront caching for static assets
- [ ] Implement API response caching in CloudFront
- [ ] Use S3 Transfer Acceleration endpoints
- [ ] Compress API responses with gzip
- [ ] Minimize cross-AZ data transfer
- [ ] Consider VPC endpoints for S3 access
- [ ] Review and optimize CloudFront cache behaviors

## 🚦 Phase 9: Production Readiness (Day 9)

### 9.1 Load Testing (1 hour)
- [ ] Install load testing tool: `brew install k6`
- [ ] Create load test script `loadtest.js`:
  ```javascript
  import http from 'k6/http';
  export let options = {
    stages: [
      { duration: '2m', target: 100 },
      { duration: '5m', target: 100 },
      { duration: '2m', target: 200 },
      { duration: '5m', target: 200 },
      { duration: '2m', target: 0 },
    ],
  };
  ```
- [ ] Run baseline test: `k6 run loadtest.js`
- [ ] Document baseline metrics:
  - [ ] Requests per second
  - [ ] P95 response time
  - [ ] Error rate
- [ ] Test auto-scaling triggers
- [ ] Test rate limiting
- [ ] Identify bottlenecks
- [ ] Optimize based on results

### 9.2 Disaster Recovery (45 mins)
- [ ] Create backup strategy:
  - [ ] Daily S3 backup to different region
  - [ ] EBS snapshots (if using)
  - [ ] Database backups (if using RDS)
- [ ] Document recovery procedures:
  - [ ] Service restoration steps
  - [ ] Data recovery steps
  - [ ] Communication plan
- [ ] Test backup restoration:
  ```bash
  aws s3 sync s3://svg-ai-production/ s3://svg-ai-backup/ --source-region us-east-1 --region us-west-2
  ```
- [ ] Create runbook for common issues
- [ ] Set up multi-region failover (optional)
- [ ] Document RTO and RPO targets
- [ ] Test full disaster recovery scenario

### 9.3 Documentation (1 hour)
- [ ] Create operations runbook:
  - [ ] Deployment procedures
  - [ ] Rollback procedures
  - [ ] Troubleshooting guide
  - [ ] Contact information
- [ ] Document architecture:
  - [ ] System diagram
  - [ ] Data flow diagram
  - [ ] Network topology
- [ ] Create API documentation:
  - [ ] Endpoint descriptions
  - [ ] Request/response examples
  - [ ] Error codes
- [ ] Write monitoring guide:
  - [ ] Dashboard explanations
  - [ ] Alarm response procedures
  - [ ] Performance baselines
- [ ] Create security documentation:
  - [ ] Security controls
  - [ ] Incident response plan
  - [ ] Compliance requirements

### 9.4 Final Validation (45 mins)
- [ ] Verify all endpoints work via CloudFront
- [ ] Test file upload → conversion → download flow
- [ ] Test AI classification and conversion
- [ ] Verify monitoring dashboards populated
- [ ] Check all alarms configured correctly
- [ ] Validate auto-scaling works
- [ ] Test rollback procedure
- [ ] Verify backups running
- [ ] Check security scans pass
- [ ] Validate costs within budget
- [ ] Performance meets SLA requirements

## 🎯 Phase 10: Go-Live (Day 10)

### 10.1 Pre-Launch Checklist (30 mins)
- [ ] All tests passing (unit, integration, load)
- [ ] Security scan completed and issues resolved
- [ ] Backups verified working
- [ ] Monitoring dashboards active
- [ ] Alarms tested and working
- [ ] Documentation complete and reviewed
- [ ] Team trained on operations procedures
- [ ] Support contact information updated
- [ ] Legal/compliance review complete
- [ ] Communication plan ready

### 10.2 DNS Cutover (30 mins)
- [ ] Lower TTL on current DNS records (24 hours before)
- [ ] Create CloudFront distribution for old site (fallback)
- [ ] Update DNS to point to new CloudFront:
  ```bash
  aws route53 change-resource-record-sets \
    --hosted-zone-id {zone-id} \
    --change-batch file://dns-change.json
  ```
- [ ] Monitor traffic shift in CloudFront
- [ ] Verify no 4xx/5xx errors spike
- [ ] Check conversion success rate
- [ ] Monitor system performance
- [ ] Keep old system running in parallel

### 10.3 Launch Communication (20 mins)
- [ ] Send launch notification to team
- [ ] Update status page (if exists)
- [ ] Post on social media (if applicable)
- [ ] Notify key customers
- [ ] Update documentation site
- [ ] Submit to monitoring services (e.g., StatusPage)

### 10.4 Post-Launch Monitoring (2 hours)
- [ ] Monitor real-time metrics dashboard
- [ ] Check CloudWatch alarms every 15 minutes
- [ ] Review X-Ray traces for errors
- [ ] Monitor auto-scaling behavior
- [ ] Check cost tracking
- [ ] Verify backup job runs successfully
- [ ] Review security alerts
- [ ] Check user feedback channels
- [ ] Document any issues encountered
- [ ] Create tickets for improvements

### 10.5 First Week Operations (ongoing)
- [ ] Daily health check review
- [ ] Review CloudWatch Insights for patterns
- [ ] Optimize based on real usage:
  - [ ] Adjust auto-scaling thresholds
  - [ ] Tune cache settings
  - [ ] Optimize container resources
- [ ] Address user feedback
- [ ] Fix any bugs discovered
- [ ] Update documentation based on learnings
- [ ] Plan first iteration improvements

## 📊 Success Metrics Tracking

### Performance KPIs (measure daily)
- [ ] Uptime percentage (target: 99.9%)
- [ ] API response time P50 (target: < 100ms)
- [ ] API response time P95 (target: < 200ms)
- [ ] API response time P99 (target: < 500ms)
- [ ] Conversion success rate (target: > 98%)
- [ ] AI model accuracy (target: > 95%)

### Business Metrics (measure weekly)
- [ ] Number of conversions per day
- [ ] Average file size processed
- [ ] User retention rate
- [ ] Cost per conversion
- [ ] AWS monthly spend vs budget

### Operational Metrics (measure daily)
- [ ] Number of deployments
- [ ] Mean time to recovery (MTTR)
- [ ] Number of incidents
- [ ] Alarm noise ratio
- [ ] Backup success rate

## 🔧 Maintenance Tasks

### Daily Tasks
- [ ] Review CloudWatch dashboards
- [ ] Check for security alerts
- [ ] Verify backups completed
- [ ] Review error logs
- [ ] Check cost anomalies

### Weekly Tasks
- [ ] Review performance trends
- [ ] Update dependencies (security patches)
- [ ] Clean up old logs and data
- [ ] Review and optimize alarms
- [ ] Team sync on issues and improvements

### Monthly Tasks
- [ ] Full disaster recovery test
- [ ] Security audit
- [ ] Cost optimization review
- [ ] Capacity planning review
- [ ] Documentation updates

## 📝 Important Commands Reference

```bash
# View ECS service status
aws ecs describe-services --cluster svg-ai-cluster --services svg-ai-service

# Force new deployment
aws ecs update-service --cluster svg-ai-cluster --service svg-ai-service --force-new-deployment

# View recent logs
aws logs tail /ecs/svg-ai --follow

# Create CloudFront invalidation
aws cloudfront create-invalidation --distribution-id {dist-id} --paths "/*"

# Scale ECS service manually
aws ecs update-service --cluster svg-ai-cluster --service svg-ai-service --desired-count 5

# View ALB target health
aws elbv2 describe-target-health --target-group-arn {tg-arn}
```

---

## ✅ Deployment Complete!

Total estimated time: 10 days (80 hours)
Estimated monthly AWS cost: $150-250

Remember to:
- Keep this checklist updated
- Document any deviations
- Share learnings with the team
- Celebrate the successful deployment! 🎉