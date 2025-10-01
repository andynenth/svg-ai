# SVG-AI Operations Manual

## Deployment

### Prerequisites
- Docker and Docker Compose
- 4GB RAM minimum, 8GB recommended
- 2 CPU cores minimum, 4 recommended

### Production Deployment
```bash
# Clone repository
git clone <repository>
cd svg-ai

# Configure environment
cp .env.example .env
# Edit .env with production values

# Deploy with monitoring
docker-compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d

# Verify deployment
curl http://localhost/health
```

## Monitoring

### Key Metrics to Monitor
- Response time: <2s for Tier 1, <5s for Tier 2, <15s for Tier 3
- Memory usage: <500MB per container
- Error rate: <1%
- Queue depth: <100 pending requests

### Grafana Dashboards
- Application Performance: http://localhost:3000/d/app-performance
- System Resources: http://localhost:3000/d/system-resources
- Error Tracking: http://localhost:3000/d/error-tracking

## Maintenance

### Daily Tasks
- Check system health: `curl http://localhost/health`
- Review error logs: `docker-compose logs svg-ai | grep ERROR`
- Monitor resource usage: `docker stats`

### Weekly Tasks
- Update dependencies: Check security advisories
- Performance review: Analyze response time trends
- Capacity planning: Review usage growth

### Monthly Tasks
- Security scan: `docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image svg-ai:latest`
- Database maintenance: Clear old cache entries
- Documentation updates: Review and update operational procedures

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory by container
docker stats --no-stream

# Restart if needed
docker-compose restart svg-ai
```

#### Slow Response Times
```bash
# Check system load
docker exec svg-ai top

# Review recent errors
docker-compose logs --tail=100 svg-ai
```

#### Connection Issues
```bash
# Verify network connectivity
docker-compose exec svg-ai curl -I http://redis:6379

# Check service dependencies
docker-compose ps
```

## System Architecture

### Container Overview
- **svg-ai**: Main application container (Flask API)
- **redis**: Cache and rate limiting storage
- **nginx**: Reverse proxy and load balancer
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards

### Service Dependencies
```
nginx -> svg-ai -> redis
prometheus -> svg-ai (metrics)
grafana -> prometheus (dashboards)
```

### Data Flow
1. Client requests hit Nginx reverse proxy
2. Nginx routes to svg-ai container
3. Flask application processes conversion requests
4. Redis handles caching and rate limiting
5. Prometheus collects metrics
6. Grafana displays monitoring dashboards

## Configuration Management

### Environment Variables
Production environment variables are managed through `.env` files:

```bash
# Core application settings
FLASK_ENV=production
SECRET_KEY=<generated-secret>
WORKERS=4

# Redis configuration
REDIS_URL=redis://redis:6379

# Upload settings
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# Security settings
FLASK_DEBUG=0
```

### Configuration Files
- `docker-compose.prod.yml`: Production container orchestration
- `nginx.conf`: Reverse proxy configuration
- `monitoring/prometheus.yml`: Metrics collection configuration
- `monitoring/grafana/dashboards/`: Dashboard definitions

## Security Operations

### Security Monitoring
1. **Vulnerability Scanning**: Run weekly container scans
2. **Log Analysis**: Monitor for suspicious activity patterns
3. **Rate Limiting**: Ensure rate limits are effective
4. **Input Validation**: Verify security validation is working

### Security Incident Response
1. **Immediate Actions**:
   - Isolate affected systems
   - Stop attack vectors
   - Preserve logs for analysis

2. **Investigation**:
   - Analyze logs for attack patterns
   - Identify compromised data
   - Document incident timeline

3. **Recovery**:
   - Apply security patches
   - Update configurations
   - Restart services if needed

4. **Post-Incident**:
   - Update security procedures
   - Improve monitoring
   - Conduct lessons learned

### Security Checklist
- [ ] Regular security scans running
- [ ] Rate limiting operational
- [ ] Input validation active
- [ ] Logs being monitored
- [ ] Security updates applied
- [ ] Backup procedures tested

## Performance Optimization

### Performance Targets
- **Tier 1 (Simple)**: <2s response time, >95% SSIM
- **Tier 2 (Medium)**: <5s response time, >90% SSIM
- **Tier 3 (Complex)**: <15s response time, >85% SSIM

### Optimization Strategies
1. **Caching**: Implement aggressive caching for repeated requests
2. **Parallel Processing**: Use batch processing for multiple images
3. **Resource Limits**: Set appropriate container resource limits
4. **Load Balancing**: Scale horizontally with multiple containers

### Performance Monitoring
```bash
# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost/api/convert

# Check resource usage
docker stats --no-stream

# Analyze processing patterns
docker-compose logs svg-ai | grep "Processing time"
```

## Backup and Recovery

### Backup Strategy
1. **Configuration Backup**: Daily backup of configuration files
2. **Cache Backup**: Weekly Redis data export
3. **Log Backup**: Continuous log shipping to external storage
4. **Image Backup**: Daily Docker image snapshots

### Recovery Procedures
1. **Configuration Recovery**:
   ```bash
   # Restore configuration from backup
   cp backup/.env .env
   cp -r backup/monitoring/ monitoring/
   ```

2. **Service Recovery**:
   ```bash
   # Stop services
   docker-compose down

   # Restore from backup
   docker-compose up -d

   # Verify recovery
   curl http://localhost/health
   ```

3. **Data Recovery**:
   ```bash
   # Restore Redis data
   docker-compose exec redis redis-cli --rdb backup.rdb
   ```

## Scaling Operations

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  svg-ai:
    scale: 3  # Run 3 instances
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
```

### Load Balancer Configuration
```nginx
upstream svg-ai-backend {
    server svg-ai_1:5000;
    server svg-ai_2:5000;
    server svg-ai_3:5000;
}
```

### Capacity Planning
- Monitor resource utilization trends
- Plan scaling based on usage patterns
- Consider geographic distribution for global deployments

## Alerting Configuration

### Critical Alerts
- System down: Response time >30s
- High error rate: >5% errors
- Memory exhaustion: >90% memory usage
- Disk space: <10% free space

### Warning Alerts
- Slow responses: >5s average response time
- Moderate error rate: >1% errors
- High memory: >70% memory usage
- Queue buildup: >50 pending requests

### Alert Channels
- Email: Critical alerts to operations team
- Slack: All alerts to monitoring channel
- PagerDuty: Critical alerts with escalation
- Dashboard: Real-time status display

## Change Management

### Deployment Process
1. **Pre-deployment**:
   - Code review and approval
   - Staging environment testing
   - Backup current configuration

2. **Deployment**:
   - Rolling deployment strategy
   - Health check verification
   - Rollback plan ready

3. **Post-deployment**:
   - Monitor for issues
   - Verify functionality
   - Document changes

### Version Control
- All configuration changes tracked in Git
- Environment-specific branches
- Tagged releases for production deployments

## Emergency Procedures

### Service Outage Response
1. **Immediate Actions** (0-5 minutes):
   ```bash
   # Check service status
   curl http://localhost/health

   # Check container status
   docker-compose ps

   # Quick restart if needed
   docker-compose restart svg-ai
   ```

2. **Investigation** (5-15 minutes):
   ```bash
   # Check logs for errors
   docker-compose logs --tail=100 svg-ai

   # Check resource usage
   docker stats --no-stream

   # Check network connectivity
   docker-compose exec svg-ai ping redis
   ```

3. **Escalation** (15+ minutes):
   - Contact development team
   - Consider full system restart
   - Implement emergency maintenance page

### Recovery Scenarios
1. **Container Failure**: Automatic restart via Docker
2. **Redis Failure**: Graceful degradation, restart Redis
3. **Nginx Failure**: Direct container access, restart Nginx
4. **Complete System Failure**: Full stack restart from backups

## Contact Information

### Primary Contacts
- **Operations Lead**: operations@company.com
- **Development Team**: dev-team@company.com
- **Security Team**: security@company.com

### Escalation Matrix
1. **Level 1**: Operations team member
2. **Level 2**: Operations lead + Development lead
3. **Level 3**: CTO + Security team

### Emergency Contacts
- **24/7 Hotline**: +1-XXX-XXX-XXXX
- **Slack Channel**: #svg-ai-ops
- **PagerDuty**: svg-ai-production service

---

## Quick Reference Commands

### Daily Operations
```bash
# Health check
curl http://localhost/health

# View logs
docker-compose logs -f svg-ai

# Check resource usage
docker stats

# Restart service
docker-compose restart svg-ai
```

### Emergency Commands
```bash
# Stop all services
docker-compose down

# Start with fresh containers
docker-compose up -d --force-recreate

# Emergency maintenance mode
docker-compose -f docker-compose.maintenance.yml up -d
```

### Monitoring Commands
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test Grafana
curl http://admin:admin@localhost:3000/api/health

# Check alerting rules
curl http://localhost:9090/api/v1/rules
```