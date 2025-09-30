# ElastiCache Redis Configuration for SVG-AI

# Random password for Redis
resource "random_password" "redis" {
  length  = 32
  special = false  # Redis doesn't support all special characters
}

# Store Redis password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "redis" {
  name        = "${local.cluster_name}-redis-password"
  description = "Redis password for SVG-AI cache"

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "redis" {
  secret_id     = aws_secretsmanager_secret.redis.id
  secret_string = jsonencode({
    password = random_password.redis.result
  })
}

# Security group for ElastiCache Redis
resource "aws_security_group" "redis" {
  name_prefix = "${local.cluster_name}-redis"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for ElastiCache Redis"

  ingress {
    description     = "Redis from EKS"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-sg"
  })
}

# ElastiCache parameter group for Redis
resource "aws_elasticache_parameter_group" "redis" {
  family      = "redis7.x"
  name_prefix = "${local.cluster_name}-redis"
  description = "Custom parameter group for Redis 7.x"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  tags = local.common_tags
}

# ElastiCache replication group (Redis cluster)
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = local.cluster_name
  description                  = "Redis cache for SVG-AI application"

  node_type                    = var.redis_node_type
  port                         = 6379
  parameter_group_name         = aws_elasticache_parameter_group.redis.name

  num_cache_clusters           = var.redis_num_cache_nodes
  engine_version               = "7.0"

  # Security
  auth_token                   = random_password.redis.result
  transit_encryption_enabled   = true
  at_rest_encryption_enabled   = true

  # Network
  subnet_group_name            = aws_elasticache_subnet_group.redis.name
  security_group_ids           = [aws_security_group.redis.id]

  # Backup configuration
  automatic_failover_enabled   = var.redis_num_cache_nodes > 1
  multi_az_enabled            = var.redis_num_cache_nodes > 1
  snapshot_retention_limit     = var.backup_retention_period
  snapshot_window             = "03:00-05:00"
  maintenance_window          = "sun:05:00-sun:07:00"

  # Notification
  notification_topic_arn      = aws_sns_topic.alerts.arn

  # Auto minor version upgrade
  auto_minor_version_upgrade  = true

  # Log delivery configuration
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis"
  })
}

# CloudWatch log group for Redis slow logs
resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/elasticache/redis/${local.cluster_name}/slow-log"
  retention_in_days = var.monitoring_retention_days

  tags = local.common_tags
}

# ElastiCache user for application access
resource "aws_elasticache_user" "app_user" {
  user_id       = "${local.cluster_name}-app"
  user_name     = "svgai-app"
  access_string = "on ~* +@all -flushall -flushdb"
  engine        = "REDIS"
  passwords     = [random_password.redis.result]

  tags = local.common_tags
}

# ElastiCache user group
resource "aws_elasticache_user_group" "app_group" {
  engine        = "REDIS"
  user_group_id = "${local.cluster_name}-app-group"
  user_ids      = ["default", aws_elasticache_user.app_user.user_id]

  tags = local.common_tags
}

# SNS topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${local.cluster_name}-alerts"

  tags = local.common_tags
}

# CloudWatch alarms for Redis
resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  alarm_name          = "${local.cluster_name}-redis-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors Redis CPU utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    CacheClusterId = "${local.cluster_name}-001"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  alarm_name          = "${local.cluster_name}-redis-memory-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "85"
  alarm_description   = "This metric monitors Redis memory utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    CacheClusterId = "${local.cluster_name}-001"
  }

  tags = local.common_tags
}