# Production Runbook
## 4-Tier SVG-AI System - Operational Procedures

**Version:** 1.0
**Last Updated:** September 30, 2025
**Owner:** Operations Team
**Review Cycle:** Monthly

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Daily Operations](#daily-operations)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Incident Response](#incident-response)
5. [Maintenance Procedures](#maintenance-procedures)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Emergency Procedures](#emergency-procedures)
8. [Contact Information](#contact-information)

---

## System Overview

### Architecture Summary
The 4-Tier SVG-AI System is a production-grade microservices application deployed on Kubernetes that converts PNG images to high-quality SVG format using intelligent optimization techniques.

#### Core Components:
- **Tier 1:** Feature Extraction & Image Analysis
- **Tier 2:** Intelligent Method Selection & Routing
- **Tier 3:** Optimization Execution & Quality Validation
- **Tier 4:** Real-time Monitoring & Continuous Improvement

#### Infrastructure:
- **Platform:** Kubernetes 1.27+
- **Namespace:** `svg-ai-4tier-prod`
- **Load Balancer:** NGINX Ingress Controller
- **Database:** PostgreSQL 15 (Primary + Replica)
- **Cache:** Redis 7.x
- **Monitoring:** Prometheus + Grafana + AlertManager

### Key Performance Indicators
| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| P95 Response Time | <15s | >20s |
| System Availability | >99.9% | <99% |
| Success Rate | >95% | <90% |
| Error Rate | <2% | >5% |
| CPU Utilization | <80% | >90% |
| Memory Utilization | <85% | >95% |

---

## Daily Operations

### Morning Checks (Start of Business Day)

#### 1. System Health Verification
```bash
# Check overall system status
kubectl get pods -n svg-ai-4tier-prod

# Expected: All pods in Running state
# Alert if: Any pods in Error, CrashLoopBackOff, or Pending state
```

#### 2. Service Health Checks
```bash
# Check service endpoints
curl -H "Authorization: Bearer ${API_KEY}" \
  https://api.svg-ai.com/api/v2/optimization/health

# Expected: HTTP 200 with "healthy" status
# Alert if: HTTP 5xx errors or "unhealthy" status
```

#### 3. Database Health
```bash
# Check PostgreSQL connectivity
kubectl exec -n svg-ai-4tier-prod postgres-4tier-0 -- \
  pg_isready -h localhost -p 5432

# Check database connections
kubectl exec -n svg-ai-4tier-prod postgres-4tier-0 -- \
  psql -c "SELECT count(*) FROM pg_stat_activity;"

# Expected: Ready status, <100 active connections
```

#### 4. Cache Health
```bash
# Check Redis connectivity and memory usage
kubectl exec -n svg-ai-4tier-prod redis-4tier-0 -- \
  redis-cli info memory

# Expected: used_memory_human < 400MB (80% of 512MB limit)
```

#### 5. Review Overnight Metrics
- Access Grafana Dashboard: `https://monitoring.svg-ai.com`
- Review overnight performance trends
- Check for any alerts or anomalies
- Verify backup completion status

### Evening Checks (End of Business Day)

#### 1. Performance Summary
```bash
# Generate daily performance report
curl -H "Authorization: Bearer ${API_KEY}" \
  https://api.svg-ai.com/api/v2/optimization/metrics/daily

# Review key metrics:
# - Total conversions processed
# - Average quality scores
# - Response time percentiles
# - Error rates by type
```

#### 2. Capacity Planning
- Review resource utilization trends
- Check auto-scaling events
- Verify storage capacity remaining
- Plan for next-day capacity if needed

#### 3. Security Review
- Review security alerts from the day
- Check for unusual access patterns
- Verify no critical security updates pending

---

## Monitoring & Alerting

### Grafana Dashboards

#### Primary Dashboard: "4-Tier SVG-AI Overview"
URL: `https://monitoring.svg-ai.com/d/4tier-overview`

**Key Panels:**
- System Health Overview
- Request Rate & Success Rate
- Response Time Percentiles
- Quality Metrics Trend
- Resource Utilization
- Error Rate by Type

#### Secondary Dashboards:
- **Performance Deep Dive:** `https://monitoring.svg-ai.com/d/4tier-performance`
- **Quality Analytics:** `https://monitoring.svg-ai.com/d/4tier-quality`
- **Infrastructure Metrics:** `https://monitoring.svg-ai.com/d/4tier-infrastructure`

### Alert Categories

#### Critical Alerts (Immediate Response Required)
| Alert | Condition | Response Time | Action |
|-------|-----------|---------------|--------|
| Service Down | All pods in namespace unavailable | <5 minutes | Page on-call engineer |
| Database Unavailable | PostgreSQL connection failures | <5 minutes | Page DBA and on-call |
| High Error Rate | >10% error rate for 5+ minutes | <10 minutes | Investigate and escalate |
| Response Time Critical | P95 >30s for 10+ minutes | <15 minutes | Check resources and optimize |

#### Warning Alerts (Response Within Business Hours)
| Alert | Condition | Response Time | Action |
|-------|-----------|---------------|--------|
| High Resource Usage | CPU/Memory >90% for 15+ minutes | <30 minutes | Scale resources |
| Quality Degradation | Avg quality <80% for 30+ minutes | <1 hour | Review optimization |
| Slow Response Time | P95 >20s for 15+ minutes | <1 hour | Performance analysis |
| Cache Hit Rate Low | <70% for 1+ hour | <2 hours | Investigate cache efficiency |

#### Info Alerts (Daily Review)
- Deployment notifications
- Scaling events
- Backup completion status
- Security scan results

### Alert Response Procedures

#### 1. Alert Acknowledgment
```bash
# Acknowledge alert in AlertManager
curl -X POST https://alertmanager.svg-ai.com/api/v1/alerts \
  -d '{"status":"acknowledged","comment":"Investigating"}'
```

#### 2. Initial Assessment
- Check Grafana dashboards for context
- Review recent deployments or changes
- Verify alert is not a false positive
- Estimate impact and severity

#### 3. Communication
- Update incident status in PagerDuty
- Notify stakeholders if customer-impacting
- Document investigation progress

---

## Incident Response

### Incident Classification

#### Severity 1 (Critical)
- **Definition:** Complete service unavailability or data loss
- **Response Time:** <5 minutes
- **Escalation:** Immediate page to on-call engineer and manager
- **Communication:** Update customers within 15 minutes

#### Severity 2 (High)
- **Definition:** Partial service degradation affecting >25% of users
- **Response Time:** <15 minutes
- **Escalation:** Page on-call engineer
- **Communication:** Update customers within 1 hour

#### Severity 3 (Medium)
- **Definition:** Limited service degradation affecting <25% of users
- **Response Time:** <1 hour
- **Escalation:** Normal business hours response
- **Communication:** Update customers within 4 hours

#### Severity 4 (Low)
- **Definition:** Minor issues with workarounds available
- **Response Time:** <4 hours
- **Escalation:** During business hours
- **Communication:** Update in next scheduled communication

### Incident Response Process

#### 1. Incident Detection & Triage
```bash
# Quick system status check
kubectl get pods,svc,ingress -n svg-ai-4tier-prod

# Check recent events
kubectl get events -n svg-ai-4tier-prod --sort-by='.lastTimestamp'

# Review application logs
kubectl logs -n svg-ai-4tier-prod -l app=svg-ai-4tier-api --tail=100
```

#### 2. Initial Response Actions
- Acknowledge incident in monitoring systems
- Create incident ticket with severity classification
- Notify appropriate stakeholders
- Begin investigation and document findings

#### 3. Investigation & Diagnosis
- Review monitoring dashboards and alerts
- Check application and infrastructure logs
- Analyze recent changes or deployments
- Identify root cause and develop fix plan

#### 4. Resolution & Recovery
- Implement fix or workaround
- Verify service restoration
- Monitor for stability over 30+ minutes
- Update incident status and close when resolved

#### 5. Post-Incident Review
- Conduct post-mortem meeting within 48 hours
- Document lessons learned and action items
- Update runbooks and procedures as needed
- Implement preventive measures

---

## Maintenance Procedures

### Regular Maintenance Schedule

#### Daily
- Health check verification
- Backup status verification
- Performance metrics review
- Alert queue review

#### Weekly
- Resource utilization analysis
- Security patch assessment
- Performance trend analysis
- Capacity planning review

#### Monthly
- Full system health assessment
- Security vulnerability scan
- Disaster recovery testing
- Documentation updates
- Performance optimization review

#### Quarterly
- Major version upgrades planning
- Full disaster recovery test
- Security penetration testing
- Comprehensive performance review
- Business continuity plan review

### Deployment Procedures

#### Standard Deployment Process
1. **Pre-deployment Checklist**
   - All tests passing in CI/CD pipeline
   - Security scans completed and approved
   - Rollback plan documented and tested
   - Change approval obtained

2. **Deployment Execution**
   ```bash
   # Deploy using blue-green strategy
   python deployment/production/go_live_deployment.py \
     --environment production \
     --image-tag v2.1.0 \
     --output deployment_report.json
   ```

3. **Post-deployment Validation**
   - Health checks passing for all services
   - Performance metrics within acceptable ranges
   - No error rate increases
   - User acceptance testing passed

4. **Rollback Procedures** (if needed)
   ```bash
   # Emergency rollback to previous version
   kubectl rollout undo deployment/svg-ai-4tier-api -n svg-ai-4tier-prod
   kubectl rollout undo deployment/svg-ai-4tier-worker -n svg-ai-4tier-prod
   ```

### Database Maintenance

#### Backup Procedures
```bash
# Daily backup verification
kubectl exec -n svg-ai-4tier-prod postgres-4tier-0 -- \
  pg_dump svgai_prod | gzip > /backups/svgai_$(date +%Y%m%d).sql.gz

# Backup retention: 30 days local, 90 days off-site
```

#### Database Performance Tuning
```sql
-- Weekly maintenance queries
ANALYZE;
REINDEX DATABASE svgai_prod;
VACUUM (ANALYZE, VERBOSE);

-- Monitor connection usage
SELECT count(*) FROM pg_stat_activity;

-- Check slow queries
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC LIMIT 10;
```

### Cache Maintenance

#### Redis Maintenance
```bash
# Weekly Redis maintenance
kubectl exec -n svg-ai-4tier-prod redis-4tier-0 -- redis-cli BGREWRITEAOF

# Monitor memory usage and eviction
kubectl exec -n svg-ai-4tier-prod redis-4tier-0 -- redis-cli info memory
```

---

## Troubleshooting Guide

### Common Issues & Solutions

#### 1. High Response Times

**Symptoms:**
- P95 response times >20 seconds
- User complaints about slow performance
- High CPU/memory usage

**Investigation Steps:**
```bash
# Check resource utilization
kubectl top pods -n svg-ai-4tier-prod

# Check active requests
curl -H "Authorization: Bearer ${API_KEY}" \
  https://api.svg-ai.com/api/v2/optimization/metrics/active

# Review recent conversions
kubectl logs -n svg-ai-4tier-prod -l app=svg-ai-4tier-api | grep "conversion_time"
```

**Common Causes & Solutions:**
- **High CPU usage:** Scale up replicas or increase resource limits
- **Memory pressure:** Increase memory limits or optimize caching
- **Database slowness:** Check for slow queries, consider read replicas
- **Large image processing:** Implement size limits or async processing

#### 2. High Error Rates

**Symptoms:**
- Error rate >5% sustained
- Specific HTTP error codes (500, 503, 429)
- User reports of failed conversions

**Investigation Steps:**
```bash
# Check error distribution
kubectl logs -n svg-ai-4tier-prod -l app=svg-ai-4tier-api | grep ERROR | tail -50

# Check database connectivity
kubectl exec -n svg-ai-4tier-prod postgres-4tier-0 -- pg_isready

# Check Redis connectivity
kubectl exec -n svg-ai-4tier-prod redis-4tier-0 -- redis-cli ping
```

**Common Causes & Solutions:**
- **Database connection issues:** Check connection pool settings
- **File system full:** Clean up temporary files, increase storage
- **Rate limiting triggered:** Adjust rate limits or scale capacity
- **External service failures:** Implement circuit breakers, fallbacks

#### 3. Quality Degradation

**Symptoms:**
- Average quality scores dropping below 85%
- User complaints about output quality
- Increased manual intervention required

**Investigation Steps:**
```bash
# Check quality metrics
curl -H "Authorization: Bearer ${API_KEY}" \
  https://api.svg-ai.com/api/v2/optimization/metrics/quality

# Review optimization methods performance
kubectl logs -n svg-ai-4tier-prod -l app=svg-ai-4tier-worker | grep "optimization_result"
```

**Common Causes & Solutions:**
- **Model drift:** Retrain quality prediction models
- **Input data changes:** Analyze recent image characteristics
- **Configuration drift:** Verify optimization parameters
- **Resource constraints:** Ensure adequate CPU/memory for processing

#### 4. Pod Crashes or Restarts

**Symptoms:**
- Pods in CrashLoopBackOff state
- Frequent pod restarts
- Memory or CPU resource limit exceeded

**Investigation Steps:**
```bash
# Check pod status and events
kubectl describe pod <pod-name> -n svg-ai-4tier-prod

# Check resource usage
kubectl top pod <pod-name> -n svg-ai-4tier-prod

# Check logs for crash reasons
kubectl logs <pod-name> -n svg-ai-4tier-prod --previous
```

**Common Causes & Solutions:**
- **Out of memory:** Increase memory limits in deployment
- **Failed health checks:** Fix application health check endpoints
- **Configuration errors:** Verify ConfigMaps and Secrets
- **Image pull errors:** Check image repository access and credentials

### Performance Optimization

#### 1. Database Optimization
```sql
-- Identify slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_conversions_created_at
ON conversions(created_at);

-- Update table statistics
ANALYZE conversions;
```

#### 2. Cache Optimization
```bash
# Monitor cache hit rates
kubectl exec -n svg-ai-4tier-prod redis-4tier-0 -- \
  redis-cli info stats | grep keyspace

# Optimize cache eviction policy
kubectl exec -n svg-ai-4tier-prod redis-4tier-0 -- \
  redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

#### 3. Application Optimization
- Monitor conversion pipeline bottlenecks
- Optimize image processing algorithms
- Implement request batching for efficiency
- Use async processing for large files

---

## Emergency Procedures

### Service Outage Response

#### Complete Service Outage
1. **Immediate Actions (0-5 minutes)**
   - Acknowledge all critical alerts
   - Check infrastructure health (K8s cluster, network)
   - Verify external dependencies (DNS, load balancer)
   - Create SEV-1 incident ticket

2. **Investigation (5-15 minutes)**
   - Review recent deployments or changes
   - Check application and infrastructure logs
   - Verify database and cache availability
   - Determine scope and impact

3. **Communication (15-30 minutes)**
   - Update status page with service outage
   - Notify stakeholders and customers
   - Provide initial estimated time to resolution
   - Escalate to engineering leadership

4. **Resolution (Ongoing)**
   - Implement fix or activate disaster recovery
   - Validate service restoration
   - Monitor for stability
   - Conduct post-incident review

### Disaster Recovery Activation

#### Trigger Conditions:
- Complete data center failure
- Major infrastructure compromise
- Extended service outage (>4 hours)
- Data corruption or loss

#### DR Activation Steps:
1. **Declare Disaster**
   - Contact DR team lead
   - Activate DR communication bridge
   - Update status page with DR activation

2. **Failover to DR Site**
   ```bash
   # Activate DR Kubernetes cluster
   kubectl config use-context dr-cluster

   # Deploy application to DR environment
   kubectl apply -f deployment/kubernetes/dr-deployment.yaml

   # Update DNS to point to DR load balancer
   # (This would typically be done via DNS management system)
   ```

3. **Validate DR Services**
   - Verify all services are running in DR site
   - Test critical user journeys
   - Confirm data synchronization status
   - Update monitoring to track DR metrics

4. **Communication & Coordination**
   - Notify customers of DR activation
   - Coordinate with infrastructure teams
   - Plan primary site recovery
   - Document DR activation process

### Security Incident Response

#### Security Alert Response
1. **Initial Assessment**
   - Determine if alert indicates actual security incident
   - Classify severity level (P1-P4)
   - Isolate affected systems if needed
   - Preserve evidence for investigation

2. **Containment**
   ```bash
   # Isolate affected pods
   kubectl label pod <suspicious-pod> quarantine=true
   kubectl delete pod <suspicious-pod> -n svg-ai-4tier-prod

   # Block suspicious traffic
   kubectl apply -f security/network-policy-block.yaml
   ```

3. **Investigation**
   - Review security logs and metrics
   - Analyze attack vectors and impact
   - Work with security team on forensics
   - Document timeline and findings

4. **Recovery & Prevention**
   - Apply security patches or fixes
   - Update security policies and controls
   - Conduct security review of affected systems
   - Update incident response procedures

---

## Contact Information

### Primary Contacts

#### On-Call Engineers
- **Primary:** on-call-engineer@svg-ai.com
- **Secondary:** backup-engineer@svg-ai.com
- **Emergency:** +1-800-SVG-4TIER

#### Escalation Matrix
| Level | Role | Contact | Response Time |
|-------|------|---------|---------------|
| L1 | Operations Engineer | ops-team@svg-ai.com | <15 minutes |
| L2 | Senior Engineer | senior-eng@svg-ai.com | <30 minutes |
| L3 | Engineering Manager | eng-manager@svg-ai.com | <1 hour |
| L4 | Engineering Director | eng-director@svg-ai.com | <2 hours |

#### Specialist Teams
- **Database Team:** dba-team@svg-ai.com
- **Security Team:** security-team@svg-ai.com
- **Infrastructure Team:** infra-team@svg-ai.com
- **Quality Assurance:** qa-team@svg-ai.com

### External Contacts

#### Vendors & Partners
- **Cloud Provider Support:** Available via console or support portal
- **Monitoring Vendor:** Available via vendor support portal
- **Security Vendor:** 24/7 SOC: security-soc@vendor.com

### Communication Channels

#### Primary Channels
- **Slack:** #svg-ai-production-alerts
- **PagerDuty:** Production escalation policies configured
- **Email:** production-team@svg-ai.com

#### Emergency Channels
- **Incident Bridge:** +1-800-INCIDENT
- **Emergency Slack:** #svg-ai-emergency
- **Status Page:** https://status.svg-ai.com

---

## Appendix

### Useful Commands

#### Kubernetes Operations
```bash
# Scale deployment
kubectl scale deployment svg-ai-4tier-api --replicas=5 -n svg-ai-4tier-prod

# Check rollout status
kubectl rollout status deployment/svg-ai-4tier-api -n svg-ai-4tier-prod

# Get detailed pod information
kubectl describe pod <pod-name> -n svg-ai-4tier-prod

# Execute commands in pod
kubectl exec -it <pod-name> -n svg-ai-4tier-prod -- /bin/bash

# Port forward for debugging
kubectl port-forward pod/<pod-name> 8080:8000 -n svg-ai-4tier-prod
```

#### Log Analysis
```bash
# Follow logs for all API pods
kubectl logs -f -l app=svg-ai-4tier-api -n svg-ai-4tier-prod

# Search logs for errors
kubectl logs -l app=svg-ai-4tier-api -n svg-ai-4tier-prod | grep ERROR

# Get logs from previous pod restart
kubectl logs <pod-name> -n svg-ai-4tier-prod --previous
```

#### Performance Analysis
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n svg-ai-4tier-prod

# Check HPA status
kubectl get hpa -n svg-ai-4tier-prod

# Check service endpoints
kubectl get endpoints -n svg-ai-4tier-prod
```

### Configuration References

#### Environment Variables
- `ENVIRONMENT=production`
- `LOG_LEVEL=INFO`
- `API_WORKERS=4`
- `DATABASE_URL=postgresql://...`
- `REDIS_URL=redis://...`

#### Resource Limits
- **API Pods:** CPU: 1000m, Memory: 2Gi
- **Worker Pods:** CPU: 2000m, Memory: 4Gi
- **Database:** CPU: 1000m, Memory: 1Gi
- **Redis:** CPU: 500m, Memory: 512Mi

---

**Document Maintenance:**
This runbook should be reviewed and updated monthly. All changes should be approved by the operations team lead and version controlled in Git.

**Last Review:** September 30, 2025
**Next Review:** October 30, 2025
**Version:** 1.0