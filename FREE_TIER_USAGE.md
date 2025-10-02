# Free Tier Usage Guidelines

## Monthly Limits (Don't Exceed!)
- **EC2**: 750 hours (run 24/7 = 720 hours, 30 hours buffer)
- **S3 Storage**: 5 GB total (currently using ~6.5MB)
- **S3 Requests**: 20K GET, 2K PUT per month
- **Data Transfer**: 15 GB outbound per month
- **CloudFront**: 50 GB transfer, 2M requests per month

## Current Deployment Details
- **Instance ID**: i-09afc9b0014ddb9fb
- **EC2 IP**: 54.166.5.132
- **Domain**: svg-ai-54.166.5.132.nip.io
- **Frontend Bucket**: svg-ai-free-frontend-1759428010
- **Models Bucket**: svg-ai-free-models-1759428010
- **CloudFront**: https://d10ryirxl0b4jh.cloudfront.net

## Access Points
- **API Endpoint**: http://svg-ai-54.166.5.132.nip.io/api/
- **Health Check**: http://svg-ai-54.166.5.132.nip.io/
- **S3 Website**: http://svg-ai-free-frontend-1759428010.s3-website-us-east-1.amazonaws.com
- **CloudFront**: https://d10ryirxl0b4jh.cloudfront.net
- **SSH Access**: ssh svg-ai-free

## Current Usage Check
```bash
# Check EC2 hours this month
aws ce get-cost-and-usage --time-period Start=2025-10-01,End=2025-10-31 --granularity MONTHLY --metrics UsageQuantity --group-by Type=DIMENSION,Key=SERVICE | grep EC2

# Check S3 storage
aws s3 ls s3://svg-ai-free-frontend-1759428010 --recursive --summarize
aws s3 ls s3://svg-ai-free-models-1759428010 --recursive --summarize
```

## Emergency Stop (If Approaching Limits)
```bash
# Stop instance to save hours
aws ec2 stop-instances --instance-ids i-09afc9b0014ddb9fb

# Start when needed
aws ec2 start-instances --instance-ids i-09afc9b0014ddb9fb
```

## After 12 Months
- t2.micro: $8.50/month
- S3: ~$0.10/month
- CloudFront: ~$1/month
- Total: ~$10/month

## Next Steps
1. Fix backend service to properly start and handle AI conversions
2. Download models from S3 to /opt/svg-ai/models/production/
3. Test API endpoints and frontend functionality
4. Set up SSL certificate (Let's Encrypt)
5. Monitor free tier usage monthly

✅ **Free Tier Deployment Complete - Infrastructure Ready!**
